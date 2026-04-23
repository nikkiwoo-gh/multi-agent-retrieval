from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
import os
import h5py
from tqdm import tqdm
from eval.readGTandPrint import readGT
from sklearn.metrics import average_precision_score 
import time

model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': 'tool_models/viCLIP/ViClip-InternVid-10M-FLT.pth',
    },
    'LanguageBind': {
        'clip_type': {
                'video': 'LanguageBind_Video_FT', 
                'image': 'LanguageBind_Image',
        },
        'cache_dir': 'LanguageBind/cache_dir/',
        'pretrained_ckpt': 'LanguageBind/LanguageBind_Video_FT',
        'tokenizer_cache_dir': 'LanguageBind/cache_dir/tokenizer_cache_dir',
    },
    'IITV': {
        'model_type': 'IITV',
        'pretrained': 'IITV/checkpoints/model_best.pth.match.tar',
    }
}          


def parse_json(text):
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None
def extract_action_and_reasoning(response):
    ##extract action and reasoning from the response
    action = None
    reasoning = None
    try:
        return_policy = json.loads(response)
        action = return_policy.get("action", "browse")
        reasoning = return_policy.get("reasoning", "brief reasoning of the action")
        return action, reasoning
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return_policy = json.loads(match)
                action = return_policy.get("action", "browse")
                reasoning = return_policy.get("reasoning", "brief reasoning of the action")
                return action, reasoning
            except json.JSONDecodeError:
                continue
    return None, None

def extract_action_and_reasoning_from_tags(response):
    """
    Extract the action and reasoning from the response text of a LLM.
    <think>reasoning process</think><answer>KEEP BROWSE/REFORMULATE</answer>
    """
    import re
    action = None
    reasoning = None
    
    # Try multiple patterns for answer extraction
    # Pattern 1: <answer>...</answer>
    action_pattern1 = r'<answer>(.*?)</answer>'
    action_match1 = re.search(action_pattern1, response, re.IGNORECASE | re.DOTALL)
    if action_match1:
        action = action_match1.group(1).strip()
    
    # Pattern 2: text within markdown code blocks that might contain the action
    if not action or action.lower() not in ['keep browse', 'browse', 'reformulate']:
        # Look for action keywords in the response
        action_lower_text = response.lower()
        if "keep browse" in action_lower_text or ("keep" in action_lower_text and "browse" in action_lower_text):
            action = "KEEP BROWSE"
        elif "browse" in action_lower_text and "keep" not in action_lower_text:
            action = "KEEP BROWSE"
        elif "reformulate" in action_lower_text:
            action = "REFORMULATE"
    
    # Normalize the action
    if action:
        action = action.strip()
        action_lower = action.lower()
        if "browse" in action_lower and "keep" not in action_lower:
            action = "KEEP BROWSE"
        elif "keep" in action_lower and "browse" in action_lower:
            action = "KEEP BROWSE"
        elif "reformulate" in action_lower:
            action = "REFORMULATE"
    
    # Extract reasoning
    # Try both <think> and <think> patterns
    reasoning_pattern1 = r'<think>(.*?)</think>'
    reasoning_match1 = re.search(reasoning_pattern1, response, re.IGNORECASE | re.DOTALL)
    if reasoning_match1:
        reasoning = reasoning_match1.group(1).strip()
    else:
        reasoning_pattern2 = r'<think>(.*?)</think>'
        reasoning_match2 = re.search(reasoning_pattern2, response, re.IGNORECASE | re.DOTALL)
        if reasoning_match2:
            reasoning = reasoning_match2.group(1).strip()
        else:
            # If no <think> tags found, extract text between start and <answer> tag
            answer_tag_pos = response.find('<answer>')
            if answer_tag_pos != -1:
                reasoning_text = response[:answer_tag_pos].strip()
                # Clean up markdown artifacts
                reasoning_text = re.sub(r'```plaintext?\n?', '', reasoning_text)
                reasoning_text = re.sub(r'```\n?', '', reasoning_text)
                reasoning = reasoning_text.strip()
    
    if reasoning:
        # Clean up markdown artifacts if present
        reasoning = re.sub(r'```[^\n]*\n?', '', reasoning).strip()
    
    return action, reasoning


def extract_reformulated_query_and_reasoning_from_tags(response):
    """
    Extract the reformulated query and reasoning from the response text of a LLM.
    <reformulate>new query</reformulate><think>reasoning process</think>
    """
    reformulated_query = None
    reasoning = None
    # Try multiple patterns for answer extraction
    # Pattern 1: <reformulate>...</reformulate>
    reformulated_query_pattern1 = r'<reformulate>(.*?)</reformulate>'
    reformulated_query_match1 = re.search(reformulated_query_pattern1, response, re.IGNORECASE | re.DOTALL)
    if reformulated_query_match1:
        reformulated_query = reformulated_query_match1.group(1).strip()
    # Pattern 2: <think>...</think>
    reasoning_pattern2 = r'<think>(.*?)</think>'
    reasoning_match2 = re.search(reasoning_pattern2, response, re.IGNORECASE | re.DOTALL)
    if reasoning_match2:
        reasoning = reasoning_match2.group(1).strip()
    else:
        # If no <think> tags found, extract text between start and <reformulate> tag
        reformulate_tag_pos = response.find('<reformulate>')
        if reformulate_tag_pos != -1:
            reasoning_text = response[:reformulate_tag_pos].strip()
            # Clean up markdown artifacts
            reasoning_text = re.sub(r'```plaintext?\n?', '', reasoning_text)
            reasoning_text = re.sub(r'```\n?', '', reasoning_text)
            reasoning = reasoning_text.strip()
    return reformulated_query, reasoning



    

def top_k_indices(scores, k):
    max_len = scores.shape[0]
    k = min(max_len, k)
    indices = np.argsort(scores)[-k:][::-1]
    return list(indices)



def construct_database(database_name,rootpath,featurename):
    vid_emb_save_file = os.path.join(rootpath,database_name,'FeatureData',featurename,'video_embeddings_whole.h5')
    vid_path_save_file = os.path.join(rootpath,database_name,'FeatureData',featurename,'video_paths_whole.h5')
    video_names_save_file = os.path.join(rootpath,database_name,'FeatureData',featurename,'video_names_whole.h5')
    print(f"constructing database: {database_name}, {featurename}")
    if database_name =='msrvtt1ktest':
       testfile = os.path.join(rootpath,database_name,'TextData','MSRVTT_JSFUSION_test.csv')
       ## file head: key,vid_key,video_id,sentence
       ## sentence is the input query
       ## video_id is the answer video of the sentence
       ## use pandas to read the file
       import pandas as pd
       df = pd.read_csv(testfile)
       query_keys = df['key'].tolist()
       query_ids = df['vid_key'].tolist()
       video_names = df['video_id'].tolist()
       query_contents = df['sentence'].tolist()
       video_paths = [os.path.join(rootpath,database_name,'VideoData',ivideo_name) for ivideo_name in video_names]
    elif database_name =='msrvtt10ktest':
        testfile = os.path.join(rootpath,database_name,'TextData',database_name+'.caption.txt')
        with open(testfile, 'r') as f:
            query_contents = f.read().splitlines()
        videolistfile  = os.path.join(rootpath,database_name,'VideoSets',database_name+'.txt')
        if not os.path.exists(videolistfile):
            raise ValueError(f"Video list file not found: {videolistfile}")
        with open(videolistfile, 'r') as f:
            video_names = f.read().splitlines()
    elif database_name =='v3c1' or database_name =='v3c2' or database_name =='v3c3' or database_name =='iacc.3':
        testfile = os.path.join(rootpath,database_name,'TextData',database_name+'.caption.txt')
        with open(testfile, 'r') as f:
            queries = f.read().splitlines()
        query_ids = [query.split(' ',1)[0] for query in queries]
        query_contents = [query.split(' ',1)[1] for query in queries]
        query_keys = query_ids
        videolistfile  = os.path.join(rootpath,database_name,'VideoSets',database_name+'.txt')
        if not os.path.exists(videolistfile):
            raise ValueError(f"Video list file not found: {videolistfile}")
        with open(videolistfile, 'r') as f:
            video_names = f.read().splitlines()
        print(f"dataset summary:\n dataset name: {database_name} \n  #video names: {len(video_names)}\n")
    queries=list(zip(query_ids,video_names,query_contents,query_keys))
    print('load #queries: ',len(queries))
    if os.path.exists(vid_emb_save_file):
        with h5py.File(vid_emb_save_file, 'r') as f:
            video_embedding_arrays = f['video_embeddings'][:]
        with h5py.File(vid_path_save_file, 'r') as f:
            video_paths = f['video_paths'][:]
        with h5py.File(video_names_save_file, 'r') as f:
            video_names = f['video_names'][:]
        
        video_names_new = []
        video_paths_new = []
        for video_name in video_names:
            video_names_new.append(video_name.decode('utf-8'))
        video_names = video_names_new
        tqdm1 = tqdm(len(video_paths), desc='Loading video paths')
        for video_path in video_paths:
            tqdm1.update(1)
            video_path = video_path.decode('utf-8')
            video_name = video_path.split('/')[-1].split('.')[0]
            if database_name =='v3c2' and not os.path.exists(video_path) :
                video_path = os.path.join(rootpath,database_name,'VideoData','V3C2_'+video_name+'.webm')
            video_paths_new.append(video_path)
        video_paths = video_paths_new
        print(f"loaded database from {vid_emb_save_file}, {vid_path_save_file}, {video_names_save_file}")
        print(f"database summary:\n dataset name: {database_name} \n #video names: {len(video_names)}\n")
    else:
        video_paths = []
        if featurename=='viclip_vid_feature' or featurename=='LanguageBind_videoLevelFeature' or featurename=='BLIP2_vidFeature':
            feature_dim = 768
        elif featurename=='mean_CLIP_ViT-B_32_vidFeature':
            feature_dim = 512
        elif featurename=='Improved_ITV':
            feature_dim = 2048
        else:
            raise ValueError(f"Feature name not supported: {featurename}")
        
        video_embedding_arrays = np.zeros((len(video_names), feature_dim))
        tqdm1 = tqdm(video_names, desc='Constructing database')
        for ivideo_idx,ivideo_name in enumerate(video_names):
            tqdm1.update(1)
            video_path = os.path.join(rootpath,database_name,'VideoData',ivideo_name+'.mp4')
            video_names[ivideo_idx] = ivideo_name
            if not os.path.exists(video_path):
                video_path = os.path.join(rootpath,database_name,'VideoData',ivideo_name+'.webm')
                if database_name =='v3c2':
                    video_path = os.path.join(rootpath,database_name,'VideoData','V3C2_'+ivideo_name+'.webm')
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                video_paths.append(None)
            else:
                video_paths.append(video_path)
            feature_path = os.path.join(rootpath,database_name,'FeatureData',featurename,'npy',ivideo_name+'.npy')
            if database_name =='v3c2':
                feature_path = os.path.join(rootpath,database_name,'FeatureData',featurename,'npy','V3C2_'+ivideo_name+'.npy')
            if not os.path.exists(feature_path):
                print(f"Feature file not found: {feature_path}")
                continue
            video_embeddings = np.load(feature_path)

            video_embedding_arrays[ivideo_idx] = video_embeddings
        with h5py.File(vid_emb_save_file, 'w') as f:
            f.create_dataset('video_embeddings', data=video_embedding_arrays)
        with h5py.File(vid_path_save_file, 'w') as f:
            f.create_dataset('video_paths', data=video_paths)
        with h5py.File(video_names_save_file, 'w') as f:
            f.create_dataset('video_names', data=video_names)
        print(f"saved database to {vid_emb_save_file}, {vid_path_save_file}, {video_names_save_file}")
    database = {'video_paths': video_paths, 'video_embeddings': video_embedding_arrays, 'video_names': video_names}
    
    if featurename == "Improved_ITV":
        ##load concept bank and concept invert file path
        concept_bank_file = os.path.join(rootpath,database_name,'FeatureData',featurename,'concept_bank.txt')
        with open(concept_bank_file, 'r') as f:
            concept_bank = f.read().splitlines()
        concept_invert_path = os.path.join(rootpath,database_name,'FeatureData',featurename,'video_concept_prob_invertfiles')
        database['concept_bank'] = concept_bank
        database['concept_invert_path'] = concept_invert_path
    return database,queries


def get_database_name(query_set):
    if query_set =='tv16' or query_set =='tv17' or query_set =='tv18':
        return 'iacc.3'
    elif query_set =='tv19' or query_set =='tv20' or query_set =='tv21':
        return 'v3c1'
    elif query_set =='tv22' or query_set =='tv23' or query_set =='tv24':
        return 'v3c2'
    else:
        raise ValueError(f"Query set not supported: {query_set}")
    
def readQuerySet(database_name):
    queryset2query_ids = {}
    queryset2gt = {}
    if database_name =='v3c1':
        query_sets = ['tv19','tv20','tv21']
    elif database_name =='v3c2':
        query_sets = ['tv22','tv23','tv24']
    elif database_name =='iacc.3':
        query_sets = ['tv16','tv17','tv18']
    for query_set in query_sets:
            
        if query_set =='tv19':
            tmp_query_ids = range(611,641)
        elif query_set =='tv20':
            tmp_query_ids = range(641,661)
        elif query_set =='tv21':
            tmp_query_ids = range(661,681)
        elif query_set =='tv22':
            tmp_query_ids = range(701,731)
        elif query_set =='tv23':
            tmp_query_ids = range(731,751)
        elif query_set =='tv24':
            tmp_query_ids = range(751,771)
        elif query_set =='tv16':
            tmp_query_ids = range(501,531)
        elif query_set =='tv17':
            tmp_query_ids = range(531,561)
        elif query_set =='tv18':
            tmp_query_ids = range(561,591)
        else:
            raise ValueError(f"Query set not supported: {query_set}")
        queryset2query_ids[query_set] = tmp_query_ids
        gt = readGT(query_set,database_name)
        queryset2gt[query_set] = gt
    return queryset2query_ids,queryset2gt


def AVS_eval_ranklist(ranklist,gt,fp,unjudge_dict,k=10):
    """
    AVS juhas mulitple gt video names for each query
    compute the rank of the gt video in the ranklist
    return the recall@1,5,10
    """
    ##filter the ranklist, except ranklist[0] is the queryid, others should have the format 'shot%05d_%d'
    query_id = ranklist[0]
    if not query_id.startswith('shot'):
        del ranklist[0]
    orig_ranklist = ranklist.copy()
    if len(ranklist)<k:
        k = len(ranklist)
    ranklist = ranklist[:k]
    ##compute the precision, recall, map@k
    match = np.zeros(len(ranklist))
    unjudge_num = 0
    unmatch_num = 0
    match_num = 0
    for i_rank,video_name in enumerate(ranklist):
        if video_name in gt:
            match[i_rank] = 1
            match_num += 1
        elif video_name in fp:
            unmatch_num += 1
        else:
            unjudge_num += 1
    if len(ranklist)==0:
        return 0,0,0,0,0,0,k
    precision = match_num/len(ranklist)
    recall = match_num/len(gt)
    # For MAP calculation, use decreasing relevance scores (higher rank = lower score)
    # This reflects the ranking order where items at the top are more relevant
    y_scores = [1.0 - (i / len(ranklist)) for i in range(len(ranklist))]
    map = average_precision_score(y_true=match.astype(bool), y_score=y_scores)
    return precision,recall,map,match_num,unmatch_num,unjudge_num,k
    
def msrvtt_eval_ranklist(ranklist,gt_video_name):
    """
    msrvtt just has one gt video name
    compute the rank of the gt video in the ranklist
    return the recall@1,5,10
    """
    ##compute the recall@1,5,10
    rank = 100
    for i,video_name in enumerate(ranklist):
        if video_name == gt_video_name:
            rank = i+1
            break
    recall_1 = rank if rank<=1 else 0
    recall_5 = rank if rank<=5 else 0
    recall_10 = rank if rank<=10 else 0
    return recall_1,recall_5,recall_10


def encode_BLIP2_text(query,BLIP2_server):
    import socket
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(BLIP2_server)
    client.send(query.encode('utf-8'))   
    reply = np.load('tmp/BLIP2_feature.npy')
    client.close()
    return reply

def encode_ImageBind_text(query,ImageBind_server):
    import socket
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(ImageBind_server)
    client.send(query.encode('utf-8'))
    reply = np.load('tmp/ImageBind_feature.npy')
    client.close()
    return reply


# === Server Management Functions ===
def _test_server_connection(server_path):
    """Test if a server socket is responsive"""
    try:
        import socket
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(2)  # 2 second timeout
        client.connect(server_path)
        client.close()
        return True
    except:
        return False

def start_servers(BLIP2_server, ImageBind_server, BLIP2_feature_file, ImageBind_feature_file, server_device):
    """Start BLIP2 and ImageBind servers independently"""
    import subprocess
    import time
    import signal
    import os
    import sys
    
    server_processes = []
    
    def start_server(server_path, server_script, max_retries=3):
        """Start a socket server with retries"""
        for attempt in range(max_retries):
            try:
                # Check if server socket already exists and is responsive
                if os.path.exists(server_path):
                    if _test_server_connection(server_path):
                        print(f"✅ {server_script} server already running and responsive")
                        return True
                    else:
                        print(f"⚠️  Removing stale socket: {server_path}")
                        os.unlink(server_path)
                
                print(f"🚀 Starting {server_script} server (attempt {attempt + 1}/{max_retries})...")
                
                # Start server in background with correct conda environment
                if server_script == 'BLIP2_text_encoder_server.py':
                    cmd = ['conda', 'run', '-n', 'lavis', 'python', server_script, f'--device={server_device}', f'--BLIP2_server={BLIP2_server}', f'--BLIP2_feature_file={BLIP2_feature_file}']
                elif server_script == 'Imagebind_text_encoder_server.py':
                    cmd = ['conda', 'run', '-n', 'imagebind', 'python', server_script, f'--device={server_device}', f'--ImageBind_server={ImageBind_server}', f'--ImageBind_feature_file={ImageBind_feature_file}']
                else:
                    cmd = [sys.executable, server_script, f'--device={server_device}', f'--BLIP2_server={BLIP2_server}', f'--BLIP2_feature_file={BLIP2_feature_file}', f'--ImageBind_server={ImageBind_server}', f'--ImageBind_feature_file={ImageBind_feature_file}']
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                
                # Check if process started successfully
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"❌ Process failed immediately. stdout: {stdout.decode()}")
                    print(f"❌ Process failed immediately. stderr: {stderr.decode()}")
                    continue
                
                # Wait for server to start and test connection
                for wait_time in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    if os.path.exists(server_path) and _test_server_connection(server_path):
                        print(f"✅ {server_script} server started successfully")
                        # Track the process for cleanup
                        server_processes.append((process, server_script))
                        return True
                
                # If we get here, server didn't start properly
                print(f"❌ {server_script} server failed to start, terminating process")
                try:
                    stdout, stderr = process.communicate(timeout=5)
                    print(f"❌ Server stdout: {stdout.decode()}")
                    print(f"❌ Server stderr: {stderr.decode()}")
                except subprocess.TimeoutExpired:
                    print(f"❌ Process didn't terminate gracefully")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception as e:
                    print(f"❌ Error getting process output: {e}")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
            except Exception as e:
                print(f"❌ Error starting {server_script} server: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
        
        return False
    
    # Start BLIP2 server
    if not start_server(BLIP2_server, 'BLIP2_text_encoder_server.py'):
        raise RuntimeError("Failed to start BLIP2_text_encoder_server.py server after 3 attempts")
    
    # Start ImageBind server
    if not start_server(ImageBind_server, 'Imagebind_text_encoder_server.py'):
        raise RuntimeError("Failed to start Imagebind_text_encoder_server.py server after 3 attempts")
    
    return server_processes

def stop_servers(server_processes):
    """Stop all running servers"""
    import signal
    import os
    
    print("🧹 Cleaning up servers...")
    for process, server_name in server_processes:
        try:
            print(f"🛑 Stopping {server_name} server (PID: {process.pid})")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
            print(f"✅ {server_name} server stopped successfully")
        except Exception as e:
            print(f"⚠️  Error stopping {server_name} server: {e}")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
    
    # Clean up socket files
    socket_files = ['tmp/BLIP2.sock', 'tmp/ImageBind.sock', 'tmp/BLIP2_ViSA.sock', 'tmp/ImageBind_ViSA.sock']
    for socket_path in socket_files:
        try:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
                print(f"🗑️  Removed socket file: {socket_path}")
        except Exception as e:
            print(f"⚠️  Could not remove socket file {socket_path}: {e}")
    
    server_processes.clear()
    print("✅ Server cleanup completed")
