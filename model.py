import torch
import clip
import numpy as np
import json
import time
import signal
import sys
import atexit
from transformers import AutoTokenizer, AutoModelForCausalLM,Qwen2_5_VLForConditionalGeneration,Qwen3VLForConditionalGeneration,AutoProcessor
from InternVid.viclip import get_viclip, retrieve_text, _frame_from_video, frames2tensor, get_vid_feat, get_text_feat_dict
from qwen_vl_utils import process_vision_info
import wandb
from utils import AVS_eval_ranklist,msrvtt_eval_ranklist,model_cfgs,extract_reformulated_query_and_reasoning_from_tags,extract_action_and_reasoning
import os
from tqdm import tqdm
import random
from IITV import get_IITV,get_clip_model,encode_BLIP2_text,encode_ImageBind_text,encode_CLIP_text
import scipy.sparse as sparse

def log_timing(step_name, start_time, end_time):
    """Log timing for a specific step"""
    duration = end_time - start_time
    print(f"⏱️  {step_name}: {duration:.2f} seconds")
    if wandb.run is not None:
        wandb.log({f"time/{step_name}": duration})
    return duration

class MultiAgentRetrieval:
    def __init__(
        self,
        MLLM_model_id="Qwen/Qwen3-VL-8B-Instruct",
        database=None,
        search_model_name='IITV',
        vlm_device="cuda:1",
        search_device="cuda:1",
        examine_number=10,
        eval_k=1000,
        BLIP2_server='tmp/BLIP2.sock',
        ImageBind_server='tmp/ImageBind.sock',
        BLIP2_feature_file='tmp/BLIP2_feature.npy',
        ImageBind_feature_file='tmp/ImageBind_feature.npy',
        greedy_action=None
    ):
        # === Timing Setup ===
        timing_info = {}
        init_start_time = time.time()
        print("Initializing Multi-Agent Retrieval Model")
        print("=" * 60)
        
        self.MLLM_model_id = MLLM_model_id
        self.vlm_device = vlm_device
        self.search_device = search_device
        self.search_model_name = search_model_name
        self.examine_number = examine_number
        self.eval_k = eval_k
        self.BLIP2_server = BLIP2_server
        self.ImageBind_server = ImageBind_server
        self.BLIP2_feature_file = BLIP2_feature_file
        self.ImageBind_feature_file = ImageBind_feature_file
        # === Load Search Model ===
        search_model_start = time.time()
        if self.search_model_name == 'viclip':
            cfg = model_cfgs['viclip-l-internvid-10m-flt']
            search_model = get_viclip(cfg['size'], cfg['pretrained'])
            assert(type(search_model) == dict and search_model['viclip'] is not None and search_model['tokenizer'] is not None)
            self.search_model, self.search_model_tokenizer = search_model['viclip'], search_model['tokenizer']
            self.search_model = self.search_model.to(self.search_device)
        elif self.search_model_name=='clip':
            CLIP_model, CLIP_preprocess = get_clip_model(device=self.search_device)
            self.search_model = CLIP_model
            self.search_model_tokenizer = CLIP_preprocess
        elif self.search_model_name == 'IITV_embedding' or self.search_model_name == 'IITV':
            cfg = model_cfgs['IITV']
            IITV_model = get_IITV(cfg['pretrained'],device=self.search_device)
            CLIP_model, CLIP_preprocess = get_clip_model(device=self.search_device)
            
            # Ensure tmp directory exists
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
            
            # Servers will be started externally
            
            self.CLIP_model = CLIP_model
            self.CLIP_preprocess = CLIP_preprocess
            self.IITV = IITV_model.to(self.search_device)
            self.concept_invert_path = database['concept_invert_path']
            self.concept_bank = database['concept_bank']
        search_model_end = time.time()
        timing_info["Search Model Loading"] = log_timing("Search Model Loading", search_model_start, search_model_end)

        # === Load Video Embeddings ===
        video_embeddings_start = time.time()
        self.total_video_number = len(database['video_paths'])
        self.dataset_video_embeddings = torch.from_numpy(database['video_embeddings'])
        self.dataset_video_ids = database['video_names']
        self.whole_video_ids = database['video_names']
        self.whole_video_paths = database['video_paths']
        self.dataset_video_paths = {}
        tqdm1 = tqdm(database['video_paths'], desc='Loading video paths')
        for i, videopath in enumerate(database['video_paths']):
            video_name = videopath.split('/')[-1]
            video_name  = video_name.split('shot')[-1]
            video_name = 'shot'+video_name.split('.')[0]
            self.dataset_video_paths[video_name] = videopath
            tqdm1.update(1)
        video_embeddings_end = time.time()
        timing_info["Video Embeddings Loading"] = log_timing("Video Embeddings Loading", video_embeddings_start, video_embeddings_end)

        # === Load VLM Model ===
        vlm_start = time.time()

        if self.MLLM_model_id.find("Qwen2.5-VL-7B-Instruct")>-1:
            self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.MLLM_model_id, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.vlm_device,
            )
        elif self.MLLM_model_id.find("Qwen3-VL-8B-Instruct")>-1:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.MLLM_model_id, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.vlm_device,
            )
        self.vlm_processor = AutoProcessor.from_pretrained(self.MLLM_model_id)
        vlm_end = time.time()
        timing_info["VLM Model Loading"] = log_timing("VLM Model Loading", vlm_start, vlm_end)
        
        # === Initialize State ===
        self.harvest_rank_list = []
        self.unmatch_rank_list = []
        self.query = None
        self.action_history = []
        self.query_performance_memory_bank = {}       ##record the query and its precision，map

        # === Initialization Summary ===
        init_end_time = time.time()
        total_init_time = init_end_time - init_start_time
        print(f"\nTiming breakdown of initialization:")
        print("=" * 60)
        for step, duration in timing_info.items():
            percentage = (duration / total_init_time) * 100 if total_init_time > 0 else 0
            print(f"{step:35s}: {duration:8.2f}s ({percentage:5.1f}%)")
        print("-" * 60)
        print(f"{'TOTAL INITIALIZATION':35s}: {total_init_time:8.2f}s (100.0%)")
        print("=" * 60)
        
        
        
    
    
       
    def retrieve_videos(self, query_text, top_k=5):
        retrieval_start = time.time()
        
        # === Text Embedding Generation ===
        embedding_start = time.time()
        if self.search_model_name == 'viclip':
            with torch.no_grad():
                query_feat = self.search_model.get_text_features(query_text, self.search_model_tokenizer, {})
                query_feat = query_feat.cpu()
        elif self.search_model_name == 'clip':
            with torch.no_grad():
                max_chars = 300
                if len(query_text) > max_chars:
                    query_text = query_text[:max_chars]
                query_feat = encode_CLIP_text(self.search_model, query_text, self.search_device)
                query_feat = torch.from_numpy(query_feat).to(self.search_device)
        elif self.search_model_name == 'IITV_embedding' or self.search_model_name == 'IITV':
            with torch.no_grad():
                BLIP2_feat = encode_BLIP2_text(query_text, self.BLIP2_server,self.BLIP2_feature_file)
                CLIP_feat = encode_CLIP_text(self.CLIP_model, query_text, self.search_device)
                ImageBind_feat = encode_ImageBind_text(query_text, self.ImageBind_server,self.ImageBind_feature_file)
                query_feat = np.concatenate([BLIP2_feat, CLIP_feat, ImageBind_feat])
                query_feat = query_feat.reshape(1,-1)
                query_feat_tensor = torch.from_numpy(query_feat).to(self.search_device)
                query_feat,query_concept_prob = self.IITV.embed_txt(query_feat_tensor,sigmoid_output=True)
                query_feat = query_feat.cpu()

        
        embedding_end = time.time()
        log_timing("Text Embedding Generation", embedding_start, embedding_end)
        
        # === Similarity Computation ===
        similarity_start = time.time()
        
        if self.search_model_name == 'IITV':
            embed_sim_scores = torch.nn.functional.cosine_similarity(query_feat, self.dataset_video_embeddings, dim=1).numpy()

            threshold = 0.99
            concept_invert_path = self.concept_invert_path
            concept_bank = self.concept_bank
            query_concept_bool = query_concept_prob>=threshold
            # Convert tensor to numpy array for indexing
            query_concept_bool_np = query_concept_bool.cpu().numpy()
            ##get concept_idx from concept decoder
            concept_idx = np.where(query_concept_bool_np)[1].tolist()
            concept_selection = np.array(concept_bank)[concept_idx].tolist()
            query_concept_weight = query_concept_prob.squeeze(0).cpu().numpy()[concept_idx].tolist()
            
            ##get concept from the query , and add it to the concept_selection and concept_idx, update query_concept_weight
            direct_selection = query_text.split(' ')
            for word in direct_selection:
                if word in concept_bank:
                    word_idx = concept_bank.index(word)
                else:
                    continue
                if word not in concept_selection:
                    concept_selection.append(word)
                    concept_idx.append(word_idx)
                    query_concept_weight.append(1.0)
                    print(f"add concept: {word}")
                else:
                    query_concept_weight[concept_idx.index(word_idx)] = 1.0
                    print(f"update concept: {word}")
            ##create a sparse matrix to store the concept similarity scores
            video_concept_scores = sparse.lil_matrix((self.total_video_number,len(concept_idx)))
            
            # Convert to numpy array for np.isin() compatibility
            current_dataset_video_ids_array = np.array(self.dataset_video_ids)
            whole_video_ids_array = np.array(self.whole_video_ids)
            tqdm2 = tqdm(len(concept_idx), desc='Loading concept invert files')
            for idconcept_idx,iconcept_idx in enumerate(concept_idx):
                tqdm2.update(1)
                concept_invert_file = os.path.join(self.concept_invert_path,str(iconcept_idx)+'.txt')
                video_idx_sorted_file = os.path.join(self.concept_invert_path,str(iconcept_idx)+'_video_idx.npy')
                video_names_sorted_file = os.path.join(self.concept_invert_path,str(iconcept_idx)+'_video_names.npy')
                video_scores_sorted_file = os.path.join(self.concept_invert_path,str(iconcept_idx)+'_video_scores.npy')
                video_idx_sorted = np.load(video_idx_sorted_file)
                video_names_sorted = np.load(video_names_sorted_file)
                video_scores_sorted = np.load(video_scores_sorted_file)
                
                video_concept_scores[video_idx_sorted,idconcept_idx] = video_scores_sorted*query_concept_weight[idconcept_idx]
            ##compute the cosine similarity between the query and the concept_sim_scores
            video_concept_scores = video_concept_scores.tocsr().toarray()
            concept_sim_scores = video_concept_scores.sum(axis=1)
            ##do the mapping back to the original video indices
            mask = np.isin(whole_video_ids_array, current_dataset_video_ids_array)
            concept_sim_scores = concept_sim_scores[mask]
            ##0-1 normalize the concept_sim_scores
            concept_sim_scores_norm = (concept_sim_scores - concept_sim_scores.min()) / (concept_sim_scores.max() - concept_sim_scores.min())
            embed_sim_scores_norm = (embed_sim_scores - embed_sim_scores.min()) / (embed_sim_scores.max() - embed_sim_scores.min())
            del embed_sim_scores,concept_sim_scores
            threshold = 0.5
            combined_sim_scores = threshold*embed_sim_scores_norm + (1-threshold)*concept_sim_scores_norm
            combined_indices = np.argsort(combined_sim_scores)[::-1][:self.eval_k*2]
            video_ids = [self.dataset_video_ids[i] for i in combined_indices]
            retrieval_end = time.time()
            log_timing("Concept Video Retrieval", retrieval_start, retrieval_end)
            return concept_sim_scores_norm, video_ids,combined_indices
        else:
            embed_sim_scores = torch.nn.functional.cosine_similarity(query_feat, self.dataset_video_embeddings, dim=1)
            embed_indices = torch.topk(embed_sim_scores, k=top_k*2).indices
            embed_indices = embed_indices.numpy()
            embed_sim_scores = embed_sim_scores[embed_indices].numpy()
            video_ids = [self.dataset_video_ids[i] for i in embed_indices]
            similarity_end = time.time()
            log_timing("Similarity Computation & Ranking", similarity_start, similarity_end)
            return embed_sim_scores, video_ids,embed_indices

    def compute_entropy(self, scores):
        probs = np.array(scores)
        probs = probs / probs.sum()
        return float(-np.sum(probs * np.log(probs + 1e-9)))

    def compute_diversity(self, scores):
        return float(1.0 - np.std(scores))

    def build_observation_prompt(self, query, eval_summary, entropy, diversity):
        return f"""Observation:
                    Query: {query}
                    TopK eval summary: {eval_summary}
                    Are the current rank list looks good? 
                    1) Answer with "browse" if you think the current rank list is good and it is worth browsing the rest of current rank list.
                    2) Answer with "reformulate" if you think the current rank list is not good and it is worth reformulating the query to refresh the rank list.
                    Answer with only one word: "browse" or "reformulate"."""
    def build_reasoning_observation_prompt(self, query, eval_summary, entropy, diversity):
        return f"""Observation:
                    Query: {query}
                    TopK eval summary: {eval_summary}
                    Are the current rank list looks good? 
                    1) Answer with "browse" if you think the current rank list is good and it is worth browsing the rest of current rank list.
                    2) Answer with "reformulate" if you think the current rank list is not good and it is worth reformulating the query to refresh the rank list.
                    output the action and reasoning in json format: {{"action": "BROWSE" or "REFORMULATE","reasoning": "brief reasoning of the action"}}."""

    def reformulate_query_with_action_reasoning(self, query,original_query, action_decision_reasoning):
        reformulate_start = time.time()
        
        
        # === Conversation Setup ===
        conversation_setup_start = time.time()
        message = f"""You are a helpful query reformulator.  Reformulate the query to have bigger chance of matching the target videos. 
        You will provide some queries and their precision obtained from the search model. 
        Besides, the reason why need to reformulate the query is also provided. 
        You can refer to these informations to help you reformulate the query.
        Suggestion: If the precision of the query is high, you are suggested to make small change (such as make it more specific) or keep the original query. If the precision of the query is low, you can imagine corresponding scenarios and describe them in the reformulated query and remember to keep the original meaning of the user query.
        Important instructions:
        - **Do not use negation words** (e.g., "not", "no", "without") — the search model performs poorly with negation.
        - Reformulate the query to maximize **semantic match** with target videos.
        - **Preserve the original meaning** of the user query.
        - The reformulated query should be **less than 30 words**.
        INPUT:
        The query and its precision obtained from the search model at each step are as follows: {str(self.query_performance_memory_bank)}
        The reason why need to reformulate the query is: {action_decision_reasoning}
        The user query is: {original_query}.
        The current search query is: {query}.
        provide your reformulated query (no more than 30 words) and reasoning process in the following format:
        <think>
        Your reasoning process: what changed, why, and how it improves matching.
        </think>
        <reformulate>
        [Reformulated query here]
        </reformulate>
        
        """
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": message},
                ]
            }
        ]
        conversation_setup_end = time.time()
        log_timing("Conversation Setup", conversation_setup_start, conversation_setup_end)
        
        # === VLM Processing ===
        vlm_processing_start = time.time()
        text = self.vlm_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.vlm_device)
        vlm_processing_end = time.time()
        log_timing("VLM Input Processing", vlm_processing_start, vlm_processing_end)
        
        # === VLM Generation ===
        vlm_generation_start = time.time()
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=1280)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        vlm_generation_end = time.time()
        log_timing("VLM Generation", vlm_generation_start, vlm_generation_end)
        
        reformulate_end = time.time()
        log_timing("Total Query Reformulation", reformulate_start, reformulate_end)
        return output_text[0],text

    def reformulate_query_normal(self, query,original_query):
        reformulate_start = time.time()
        
        
        # === Conversation Setup ===
        conversation_setup_start = time.time()
        message = f"""You are a helpful query reformulator.  Reformulate the query to have bigger chance of matching the target videos. 
        INPUT:
        The user query is: {original_query}.
        The current search query is: {query}.
        output the reformulated query (no more than 30 words) in the following format:
        <reformulate>
        [Reformulated query here]
        </reformulate>
        
        """
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": message},
                ]
            }
        ]
        conversation_setup_end = time.time()
        log_timing("Conversation Setup", conversation_setup_start, conversation_setup_end)
        
        # === VLM Processing ===
        vlm_processing_start = time.time()
        text = self.vlm_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.vlm_device)
        vlm_processing_end = time.time()
        log_timing("VLM Input Processing", vlm_processing_start, vlm_processing_end)
        
        # === VLM Generation ===
        vlm_generation_start = time.time()
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=1280)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        vlm_generation_end = time.time()
        log_timing("VLM Generation", vlm_generation_start, vlm_generation_end)
        
        reformulate_end = time.time()
        log_timing("Total Query Reformulation", reformulate_start, reformulate_end)
        return output_text[0],text


    def vlm_score(self, query, rank_list,video_paths):
        vlm_scoring_start = time.time()
        eval_scores = np.zeros(len(rank_list))
        fail_read_video_ids = []
        
        print(f"🎬 Starting VLM scoring for {len(rank_list)} videos")
        
        for i, video_id in enumerate(rank_list):
            video_scoring_start = time.time()
            video_path = video_paths[video_id]
            prompt =f"""You are a helpful assistant that can judge whether the video is relevant to the query.
            The query is {query}.
            Please judge whether the video is relevant to the query.
            answer with only one word: "match" or "unmatch".
            """
            conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ]
            try:
                # === VLM Processing ===
                vlm_processing_start = time.time()
                text = self.vlm_processor.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                image_inputs, video_inputs = process_vision_info(conversation)
                inputs = self.vlm_processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                inputs = inputs.to(self.vlm_device)
                vlm_processing_end = time.time()
                
                # === VLM Generation ===
                vlm_generation_start = time.time()
                generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=1280)
                generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                output_text = self.vlm_processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                vlm_generation_end = time.time()
                
                match_score = 1 if output_text[0] == "match" else 0
                eval_scores[i] = match_score
                
                video_scoring_end = time.time()
                video_duration = video_scoring_end - video_scoring_start
                print(f"  Video {i+1}/{len(rank_list)} ({video_id}): {video_duration:.2f}s - {'✅ MATCH' if match_score else '❌ UNMATCH'}")
                
            except Exception as e:
                fail_read_video_ids.append(video_id)
                print(f"❌ Error processing video {video_id}: {e}")
                continue
        
        # === Summary Generation ===
        summary_start = time.time()
        match_videos_ids = [video_id for i, video_id in enumerate(rank_list) if eval_scores[i] == 1]
        unmatch_videos_ids = [video_id for i, video_id in enumerate(rank_list) if eval_scores[i] == 0]
        eval_summary = {
            'total_number_of_videos': len(eval_scores),
            'match_videos_number': np.sum(eval_scores),
            'unmatch_videos_number': len(eval_scores) - np.sum(eval_scores),
        }
        summary_end = time.time()
        log_timing("VLM Scoring Summary", summary_start, summary_end)
        
        vlm_scoring_end = time.time()
        log_timing("Total VLM Scoring", vlm_scoring_start, vlm_scoring_end)
        
        print(f"📊 VLM Scoring Results: {eval_summary['match_videos_number']}/{eval_summary['total_number_of_videos']} matches")
        return eval_summary, match_videos_ids, unmatch_videos_ids,fail_read_video_ids
    def update_search_space(self, examined_video_ids,indices,video_ids):
        # Create mapping for O(1) lookup instead of O(n) list.index()
        video_id_to_j = {video_id: j for j, video_id in enumerate(video_ids)}
        
        remove_indices = []
        for video_id in examined_video_ids:
            if video_id in video_id_to_j:
                j = video_id_to_j[video_id]
                rank = indices[j]
                remove_indices.append(rank)
        
        cur_length = len(self.dataset_video_embeddings)
        remaining_indices = set(range(cur_length)) - set(remove_indices)
        remaining_indices = list(remaining_indices)
        self.dataset_video_embeddings = self.dataset_video_embeddings[remaining_indices]
        self.dataset_video_ids = [self.dataset_video_ids[i] for i in remaining_indices]
        self.dataset_video_paths = {video_id: self.dataset_video_paths[video_id] for video_id in self.dataset_video_ids }
        #log remain video number
        wandb.log({
            "remain_video_number": len(self.dataset_video_embeddings),
        })
        
    def run_policy_mllm(self,prompt):
        policy_start = time.time()
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        # === VLM Processing ===
        vlm_processing_start = time.time()
        text = self.vlm_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.vlm_device)
        vlm_processing_end = time.time()
        log_timing("Run Policy with MLLM", vlm_processing_start, vlm_processing_end)
        
        # === VLM Generation ===
        vlm_generation_start = time.time()
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=1280)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        vlm_generation_end = time.time()
        log_timing("Run Policy with MLLM Generation", vlm_generation_start, vlm_generation_end)
        
        policy_end = time.time()
        log_timing("Total Run Policy with MLLM", policy_start, policy_end)
        return output_text[0]
    
    def run(self, original_query, max_steps=3, top_k=5,log_file=None,gt=None,fp=None,unjudge_dict=None, greedy_action=None,action_type='normal',reformulation_type='normal'):
        # === Run Setup ===
        run_start_time = time.time()
        print(f"\n🚀 Starting ViSA Zero-Shot Run")
        print(f"Query: {original_query}")
        print(f"Max Steps: {max_steps}, Top-K: {top_k}")
        print("=" * 80)
        
        query = original_query
        self.query = query
        self.gt = gt
        self.fp = fp
        self.unjudge_dict = unjudge_dict
        fail_read_video_ids = []
        cur_rank_list = {}
        
        # === Initial Video Retrieval ===
        initial_retrieval_start = time.time()
        sim_scores, video_ids,indices = self.retrieve_videos(query, self.eval_k)
        sim_scores_norm = (sim_scores - sim_scores.min() )/ (sim_scores.max() - sim_scores.min())
        cur_rank_list['sim_scores'] =sim_scores_norm
        cur_rank_list['video_ids'] = video_ids
        cur_rank_list['video_indices'] = indices
        initial_retrieval_end = time.time()
        log_timing("Initial Video Retrieval", initial_retrieval_start, initial_retrieval_end)
        examine_window_start = 0
        examine_window_end = self.examine_number
        ##log new search result
        self.log_result(log_file,"initial_search", {"video_ids": video_ids[:10],'query': query})
        # === Main Loop ===
        step_timings = []
        for step in range(max_steps):
            step_start_time = time.time()
            print(f"\n🔁 Step {step + 1}/{max_steps}")
            print("-" * 60)
            
            # === Video Examination ===
            examination_start = time.time()
            examined_video_ids = cur_rank_list['video_ids'][:self.examine_number]
            examined_video_paths = {video_id: self.dataset_video_paths[video_id] for video_id in examined_video_ids if video_id in self.dataset_video_paths}
            self.update_search_space(examined_video_ids,cur_rank_list['video_indices'],cur_rank_list['video_ids'])

            self.eval_cur_step(log_file,examined_video_ids,step)
            examination_end = time.time()
            log_timing(f"Video Evaluation (GT) (Step {step+1})", examination_start, examination_end)

            # === VLM Scoring ===
            vlm_scoring_start = time.time()
            cur_sim_scores = sim_scores_norm[:self.examine_number]
            self.sim_scores_norm = cur_sim_scores
            vlm_eval_summary, match_videos_ids, unmatch_videos_ids,fail_load_videos = self.vlm_score(query, examined_video_ids,examined_video_paths)
            fail_read_video_ids = fail_read_video_ids + fail_load_videos
            vlm_scoring_end = time.time()
            log_timing(f"VLM Scoring (Step {step+1})", vlm_scoring_start, vlm_scoring_end)
            if self.query not in self.query_performance_memory_bank:
                keyname = f"step_{step}_query_{self.query}"
                self.query_performance_memory_bank[keyname] = {}
                self.query_performance_memory_bank[keyname]['precision'] = len(match_videos_ids)/len(examined_video_ids)
                self.query_performance_memory_bank[keyname]['ranklist_window_start'] = examine_window_start
                self.query_performance_memory_bank[keyname]['ranklist_window_end'] = examine_window_end
            # === Rank List Update ===
            rank_update_start = time.time()
            for i, video_id in enumerate(examined_video_ids):
                if video_id not in self.harvest_rank_list and video_id in match_videos_ids:
                    self.harvest_rank_list.append(video_id)
                else:
                    self.unmatch_rank_list.append(video_id)
            if 'match_videos_ids' not in cur_rank_list:
                cur_rank_list['match_videos_ids'] = match_videos_ids
            else:
                cur_rank_list['match_videos_ids'] = cur_rank_list['match_videos_ids'] + match_videos_ids
            if 'unmatch_videos_ids' not in cur_rank_list:
                cur_rank_list['unmatch_videos_ids'] = unmatch_videos_ids
            else:
                cur_rank_list['unmatch_videos_ids'] = cur_rank_list['unmatch_videos_ids'] + unmatch_videos_ids
            entropy = self.compute_entropy(sim_scores[:self.examine_number])
            diversity = self.compute_diversity(sim_scores[:self.examine_number])
            rank_update_end = time.time()
            log_timing(f"Rank List Update (Step {step+1})", rank_update_start, rank_update_end)
            
            ##log initial result
            self.log_result(log_file,"evaluation_result", {"query": self.query, "match_videos_ids": match_videos_ids, "unmatch_videos_ids": unmatch_videos_ids, "fail_load_videos": fail_load_videos, "entropy": entropy, "diversity": diversity,'examined_video_ids': examined_video_ids})
            if step == 0:
                action = 'initial'
                action_reasoning = "initial search"
            self.action_history.append({"step": step, "action": action, "reasoning": action_reasoning, "reward": len(match_videos_ids)/len(examined_video_ids)})

            if greedy_action == "browse" or greedy_action=="reformulate":
                action = greedy_action
                action_reasoning = "greedy action"
            elif greedy_action == "random":
                action = random.choice(["browse", "reformulate"])
                action_reasoning = "random action"
            else:
                # === Policy Decision ===
                policy_start = time.time()
                if action_type == 'normal':
                    prompt = self.build_observation_prompt(query, vlm_eval_summary,entropy, diversity)
                    action = self.run_policy_mllm(prompt)
                    self.log_result(log_file,"policy", {"policy": action,"action-type": action_type,"prompt": prompt})
                elif action_type == 'reasoning':
                    prompt = self.build_reasoning_observation_prompt(query, vlm_eval_summary,entropy, diversity)
                    action_response = self.run_policy_mllm(prompt)
                    action, action_reasoning = extract_action_and_reasoning(action_response)
                    self.log_result(log_file,"policy", {"policy": action_response,"action": action,"action_reasoning": action_reasoning,"action-type": action_type,"prompt": prompt})
                else:
                    raise ValueError(f"Invalid action type: {action_type}")
                policy_end = time.time()
                log_timing(f"Policy Decision (Step {step+1})", policy_start, policy_end)
            
            # Ensure action is a valid string
            if action is None:
                print("⚠️ Warning: Policy returned None, defaulting to 'browse'")
                action = "browse"
            elif not isinstance(action, str):
                print(f"⚠️ Warning: Policy returned non-string type {type(action)}, converting to string")
                action = str(action).strip().lower()
            else:
                action = action.strip().lower()
            
            print(f"🧠 Action: {action.upper()}")
            print(f"🎞️ harvest rank list: {self.harvest_rank_list}")
            print(f"🎞️ unmatch rank list: {self.unmatch_rank_list}")
            ##log action
            self.log_result(log_file,"action", {"action": action})
            
            # === Action Execution ===
            action_execution_start = time.time()
            if action == "browse":
                # ===  Video Retrieval on updated space space with current query===
                examine_window_start += self.examine_number
                examine_window_end += self.examine_number
                ##log browse
                self.log_result(log_file,"browse", {"query": self.query,"remain_video_number": len(self.dataset_video_embeddings),"examine_window_start": examine_window_start, "examine_window_end": examine_window_end})

            elif action=="reformulate":
                if reformulation_type=="normal":
                    reformulate_response,reformulate_prompt = self.reformulate_query_normal(self.query,original_query)
                elif reformulation_type=="with_action_reasoning":
                    reformulate_response,reformulate_prompt = self.reformulate_query_with_action_reasoning(self.query,original_query,action_reasoning)
                reformulated_query, reformulate_reasoning = extract_reformulated_query_and_reasoning_from_tags(reformulate_response)
                self.log_result(log_file,"reformulate", {"reformulation_type": reformulation_type,"reformulated_query": reformulated_query,'original_query': query,'previous_query': self.query,'reformulate_reasoning': reformulate_reasoning,'reformulate_prompt': reformulate_prompt})

                # Safety check: ensure reformulated_query is not None
                if reformulated_query is None:
                    print("⚠️ Warning: reformulated_query is None, using original query")
                    reformulated_query = original_query
                
                # Ensure reformulated_query is a string
                if not isinstance(reformulated_query, str):
                    print(f"⚠️ Warning: reformulated_query is not a string (type: {type(reformulated_query)}), converting to string")
                    reformulated_query = str(reformulated_query) if reformulated_query is not None else original_query
                    
                ##in case  reformulated query is too long, limit the length to 50 words
                if isinstance(reformulated_query, str) and len(reformulated_query) > 50:
                    reformulated_query = reformulated_query.split(' ')[:50]
                    reformulated_query = ' '.join(reformulated_query)
                ## in case it is empty, use the original query
                if isinstance(reformulated_query, str) and reformulated_query == "":
                    reformulated_query = original_query
                self.query = reformulated_query
                examine_window_start = 0
                examine_window_end = self.examine_number
                ##log reformulate
            
            ##update the ranklist with the updated search space
            ##1. browse (updated examine window-> implementation by search based on previous query) 
            ##2. reformulate (use reformulated query) 
            sim_scores, video_ids,indices = self.retrieve_videos(self.query, self.eval_k)
            sim_scores_norm = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min())
            cur_rank_list = {}
            cur_rank_list['sim_scores'] =sim_scores_norm
            cur_rank_list['video_ids'] = video_ids
            cur_rank_list['video_indices'] = indices
            cur_sim_scores = sim_scores_norm[:self.examine_number]
            self.sim_scores_norm = cur_sim_scores

            ##log new search result
            self.log_result(log_file,"search", {"video_ids": video_ids,'query': self.query})
            
            action_execution_end = time.time()
            log_timing(f"Action Execution (Step {step+1})", action_execution_start, action_execution_end)
            
            if len(self.harvest_rank_list) > self.eval_k:
                print(f"harvest rank list (length: {len(self.harvest_rank_list)}) reach the eval_k({self.eval_k}). Stopping.")
                break
            ##log number of harvest and unmatch videos
            wandb.log({
                "number_of_harvest_videos": len(self.harvest_rank_list),
            })
            
            # === Step Summary ===
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_timings.append(step_duration)
            self.log_result(log_file,"step_summary", {"step": step+1,"step_duration": step_duration,"number of videos harvested": len(self.harvest_rank_list),"number of videos unmatch": len(self.unmatch_rank_list)})
            print(f"⏱️  Step {step+1} Total Time: {step_duration:.2f} seconds")
            print("=" * 60)
            


        # === Run Summary ===
        run_end_time = time.time()
        total_run_time = run_end_time - run_start_time
        print(f"\n🏁 ViSA Zero-Shot Run Complete")
        print("=" * 80)
        print(f"Total Run Time: {total_run_time:.2f} seconds")
        print(f"Average Step Time: {np.mean(step_timings):.2f} seconds")
        print(f"Steps Completed: {len(step_timings)}")
        print(f"Final Harvest List Size: {len(self.harvest_rank_list)}")
        print("=" * 80)

        return self.harvest_rank_list,[]
        
    def log_result(self, log_file,step_name, result_dict):
        try:
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(f"{step_name}: {result_dict}\n")
            if step_name == "search":
                print(f"{step_name}: {result_dict['video_ids'][:10]}")
            elif step_name =="new_result":
                print(f"{step_name}: {result_dict['video_ids'][:10]}")
            else:
                print(f"{step_name}: {result_dict}")
            if step_name == "evaluation_result":
                ##log number of match and unmatch videos
                wandb.log({
                    "number_of_match_videos": len(result_dict["match_videos_ids"]),
                    "number_of_unmatch_videos": len(result_dict["unmatch_videos_ids"]),
                    "number_of_fail_load_videos": len(result_dict["fail_load_videos"]),
                    'diversity': result_dict["diversity"],
                    'entropy': result_dict["entropy"],
                })
            elif step_name == "browse":
                ##log examine window start and end
                wandb.log({
                    "examine_window_start": result_dict["examine_window_start"],
                    "examine_window_end": result_dict["examine_window_end"],
                })
                wandb.log({
                    "action": 0,  # 0 for browse
                })
            elif step_name == "reformulate":
                ##log reformulated query
                wandb.log({
                    "action": 1,  # 1 for reformulate
                })
        except Exception as e:
            print(f"Error logging result {step_name}: {e}")
            return
    def eval_cur_step(self,log_file,rank_list,step_idx):
        self.log_result(log_file,"eval_cur_step", {"rank_list": rank_list})
        precision,recall,map,match_num,unmatch_num,unjudge_num,k = AVS_eval_ranklist(rank_list,self.gt,self.fp,self.unjudge_dict,k=self.eval_k)
        wandb.log({
            "val/precision": precision,
            "val/recall": recall,
            "val/map": map,
            "val/match_num": match_num,
            "val/unmatch_num": unmatch_num,
            "val/unjudge_num": unjudge_num,
            "val/k": k,
            "val/step_idx": step_idx,
        })
        self.log_result(log_file,"eval_cur_step", {"step": step_idx,"precision": precision, "recall": recall, "map": map, "match_num": match_num, "unmatch_num": unmatch_num, "unjudge_num": unjudge_num, "k": k})
    
