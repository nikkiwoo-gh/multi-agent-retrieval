# Multi-Agent Retrieval System - Main Entry Point

from model import MultiAgentRetrieval
from utils import start_servers, stop_servers
from utils import construct_database,readQuerySet,AVS_eval_ranklist
from TRECVid_AVS_eval import TRECVid_AVS_eval

import time
import os
from tqdm import tqdm
import numpy as np
from eval.readGTandPrint import readGT
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_name', type=str, default='v3c1')
    parser.add_argument('--MLLM_model_id', type=str, default='/mnt_nas1/shared/Qwen2.5-VL-7B-Instruct/')
    parser.add_argument('--search_model_name', type=str, default='viclip')
    parser.add_argument('--featurename', type=str, default='viclip_vid_feature')
    parser.add_argument('--rootpath', type=str, default='/mnt_nas1/ABC/data/VBS_data/dataset/TRECVid/')
    parser.add_argument('--eval_k', type=int, default=100)
    parser.add_argument('--examine_number', type=int, default=20)
    parser.add_argument('--start_query_idx', type=int, default=0)
    parser.add_argument('--llama_device', type=str, default='cuda:1')
    parser.add_argument('--vlm_device', type=str, default='cuda:2')
    parser.add_argument('--search_device', type=str, default='cuda:3')
    parser.add_argument('--server_device', type=str, default='cuda:3')
    parser.add_argument('--BLIP2_server', type=str, default='/mnt_nas1/ABC/code/ViSearchAgent/tmp/BLIP2_ViSA.sock')
    parser.add_argument('--ImageBind_server', type=str, default='/mnt_nas1/ABC/code/ViSearchAgent/tmp/ImageBind_ViSA.sock')
    parser.add_argument('--BLIP2_feature_file', type=str, default='/mnt_nas1/ABC/code/ViSearchAgent/tmp/BLIP2_ViSA_feature.npy')
    parser.add_argument('--ImageBind_feature_file', type=str, default='/mnt_nas1/ABC/code/ViSearchAgent/tmp/ImageBind_ViSA_feature.npy')
    parser.add_argument('--MAX_ITER', type=int, default=200)
    parser.add_argument('--greedy_action', type=str, default=None,choices=['browse','reformulate','random','None','reformulate_as_suggestion'])
    parser.add_argument('--action_type', type=str, default='reasoning',choices=['normal','reasoning'])
    parser.add_argument('--reformulation_type', type=str, default='with_action_reasoning',choices=['normal',"with_action_reasoning"])
    return parser.parse_args()

def log_timing(step_name, start_time, end_time):
    """Log timing for a specific step"""
    duration = end_time - start_time
    print(f"⏱️  {step_name}: {duration:.2f} seconds")
    return duration

def save_eval_result(database_name,search_model_name,query_id,gt,fp,unjudge_dict,gt_file,ranklist,eval_k,eval_file,temp_file='temp_eval.txt'):
    if database_name =='v3c1' or database_name =='v3c2' or database_name =='v3c3' or database_name =='iacc.3':
        precision,recall,map,match_num,unmatch_num,unjudge_num,k = AVS_eval_ranklist(ranklist,gt,fp,unjudge_dict,k=eval_k)
        xinfAP = TRECVid_AVS_eval(query_id,gt_file,ranklist,eval_k,temp_file=temp_file)
        with open(eval_file, 'w') as writer:
            writer.write(f"search_model_name: {search_model_name}\n")
            writer.write(f"query_id: {query_id}\n")
            writer.write(f"precision: {precision}\n")
            writer.write(f"recall: {recall}\n")
            writer.write(f"map: {map}\n")
            writer.write(f"xinfAP: {xinfAP}\n")
            writer.write(f"match_num: {match_num}\n")
            writer.write(f"unmatch_num: {unmatch_num}\n")
            writer.write(f"unjudge_num: {unjudge_num}\n")
            writer.write(f"k: {k}\n")
        print(f"eval_result: {query_id,precision,recall,map,xinfAP,match_num,unmatch_num,unjudge_num,k}")
        eval_result = (search_model_name,query_id,precision,recall,map,xinfAP,match_num,unmatch_num,unjudge_num,k,str(ranklist))

        return eval_result
def save_ranklist(query_id,ranklist,savefile):
    with open(savefile, 'w') as writer:
        writer.write(f'{query_id} {ranklist}')
    return savefile

def main(database_name,MLLM_model_id,search_model_name,featurename,rootpath,eval_k,examine_number,llama_device,vlm_device,search_device,server_device,BLIP2_server,ImageBind_server,BLIP2_feature_file,ImageBind_feature_file,MAX_ITER,greedy_action,start_query_idx,action_type,reformulation_type):
    # === Timing Setup ===
    timing_info = {}
    main_start_time = time.time()
    
    print("Starting Multi-Agent Retrieval")
    print("=" * 80)
    
    # === Database Construction ===
    db_construction_start = time.time()
    database,queries = construct_database(database_name,rootpath,featurename)
    db_construction_end = time.time()
    timing_info["Database Construction"] = log_timing("Database Construction", db_construction_start, db_construction_end)
    

    # === Query Set Setup ===
    query_setup_start = time.time()
    if database_name =='v3c1' or database_name =='v3c2' or database_name =='v3c3' or database_name =='iacc.3':
        queryset2query_ids,queryset2gt = readQuerySet(database_name)
        target_query_ids = []
        queryid2queryset = {}
        for query_set in queryset2query_ids:
            target_query_ids.extend(queryset2query_ids[query_set])
            for query_id in queryset2query_ids[query_set]:
                queryid2queryset[query_id] = query_set
        precision_results = []
        recall_results = []
        map_results = []
        match_num_results = []
        unmatch_num_results = []
        unjudge_num_results = []
        k_results = []
    else:
        queryset2query_ids = None
        target_query_ids = [qurey_id for query_id in queries]
        queryid2gt = None
    query_setup_end = time.time()
    timing_info["Query Set Setup"] = log_timing("Query Set Setup", query_setup_start, query_setup_end)

    # === Directory Setup ===
    dir_setup_start = time.time()
    if greedy_action is not None:
        savepath = os.path.join(rootpath,database_name,'results','ViSA_zero_shot_'+'_MLLM_'+MLLM_model_id.split('/')[-2]+'_search_model_'+search_model_name+'_eval_k_'+str(eval_k)+'_examine_number_'+str(examine_number)+'_MAX_ITER_'+str(MAX_ITER)+'_greedy_action_'+greedy_action)
    else:
        savepath = os.path.join(rootpath,database_name,'results','ViSA_zero_shot_'+'_MLLM_'+MLLM_model_id.split('/')[-2]+'_search_model_'+search_model_name+'_eval_k_'+str(eval_k)+'_examine_number_'+str(examine_number)+'_MAX_ITER_'+str(MAX_ITER))
    
    if action_type == 'reasoning':
        savepath = savepath + '_action_type_reasoning'
    else:
        savepath = savepath


    if reformulation_type == 'with_action_reasoning':
        savepath = savepath + '_reformulation_type_with_action_reasoning'
    else:
        savepath = savepath
    
    logrootpath = os.path.join(savepath,'logs')
    captionpath = os.path.join(rootpath,database_name,'results',MLLM_model_id.split('/')[-1]+'_captions')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(logrootpath):
        os.makedirs(logrootpath)
    if not os.path.exists(captionpath):
        os.makedirs(captionpath)
    dir_setup_end = time.time()
    
    # === Query Processing Setup ===
    ##format in each line: query_id video_id1 video_id2 video_id3 video_id4 video_id5      
    query_tqdm = tqdm(enumerate(queries), desc='Processing queries')
    outlines = []
    eval_results = {}
    spend_time = []
    query_processing_start = time.time()
    
    if search_model_name == 'IITV':
        # === Server Initialization (outside query loop) ===
        server_init_start = time.time()
        server_processes = start_servers(BLIP2_server, ImageBind_server, BLIP2_feature_file, ImageBind_feature_file, server_device)
        server_init_end = time.time()
        timing_info["Server Initialization"] = log_timing("Server Initialization", server_init_start, server_init_end)
    else:
        server_processes = None
    try:
        for query_idx,(query_id,gt_video_name,query,query_key) in enumerate(queries):
            if query_idx<start_query_idx:
                continue
            if query_id.startswith('Textual') or query_id.startswith('t') or query_id.startswith('vbs'):
                continue
            if not int(query_id) in target_query_ids:
                continue
            savefile = os.path.join(savepath,f'rank_list_{query_id}.txt')
            eval_file = os.path.join(savepath,f'eval_results_{query_id}.txt')
            log_file = os.path.join(logrootpath,f'log_{query_id}_{query_key}_{query}.txt')
           
      

            # === Ground Truth Setup ===
            gt_setup_start = time.time()
            cur_query_set = queryid2queryset[int(query_id)]
            GT,FP,UNJUDGE_DICT = queryset2gt[cur_query_set]
            gt = GT['1'+str(query_id)]
            fp = FP['1'+str(query_id)]
            unjudge_dict = UNJUDGE_DICT['1'+str(query_id)]
            gt_setup_end = time.time()
            gt_file = os.path.join(rootpath,database_name,'TextData',f'avs.qrels.{cur_query_set}.{"1"+query_id}')
            
            if not os.path.exists(savefile):
                # try:
                # === Wandb Initialization ===
                wandb_init_start = time.time()
                if greedy_action is not None:
                    wandb.init(project="MultiAgentRetrieval", name="MultiAgentRetrieval_"+database_name+"_"+MLLM_model_id.split('/')[-1]+"_"+search_model_name+"_"+str(eval_k)+'_examine_number_'+str(examine_number)+'_greedy_action_'+greedy_action+'query_'+str(query_id))
                else:
                    wandb.init(project="MultiAgentRetrieval", name="MultiAgentRetrieval_"+database_name+"_"+MLLM_model_id.split('/')[-1]+"_"+search_model_name+"_"+str(eval_k)+'_examine_number_'+str(examine_number)+'query_'+str(query_id))
                wandb_init_end = time.time()
                timing_info["Wandb Initialization"] = log_timing("Wandb Initialization", wandb_init_start, wandb_init_end)

                # === Agent Initialization (per query) ===
                agent_init_start = time.time()
                agent = MultiAgentRetrieval(
                    MLLM_model_id=MLLM_model_id,
                    database=database,
                    search_model_name=search_model_name,
                    examine_number=examine_number,
                    vlm_device=vlm_device,
                    search_device=search_device,
                    BLIP2_server=BLIP2_server,
                    ImageBind_server=ImageBind_server,
                    BLIP2_feature_file=BLIP2_feature_file,
                    ImageBind_feature_file=ImageBind_feature_file,
                    eval_k=eval_k,
                    greedy_action=greedy_action
                )
                agent_init_end = time.time()
                timing_info[f"Agent Initialization (Query {query_id})"] = log_timing(f"Agent Initialization (Query {query_id})", agent_init_start, agent_init_end)

                try:
                    query_start_time = time.time()
                    print(f"\nStarting Multi-Agent Retrieval for Query {query_id}: {query}")
                    print("-" * 80)
                            
                    # === Agent Run ===
                    agent_run_start = time.time()
                    ranklist,log_buffer = agent.run(query, max_steps=MAX_ITER,top_k=examine_number,log_file=log_file,gt=gt,fp=fp,unjudge_dict=unjudge_dict,greedy_action=greedy_action,action_type=action_type,reformulation_type=reformulation_type)
                    agent_run_end = time.time()
                    agent_run_duration = log_timing(f"Agent Run (Query {query_id})", agent_run_start, agent_run_end)
                            
                    # === Result Processing ===
                    result_processing_start = time.time()
                    ranklist = [item.strip().strip('"').strip("'") for item in ranklist]
                    ##remove redundant video names
                    ranklist_new = []
                    for video_name in ranklist:
                        if video_name not in ranklist_new:
                            ranklist_new.append(video_name)
                    ranklist = ranklist_new
                    ranklist_str = ' '.join(ranklist)
                    outlines.append(f'{query_id} {ranklist_str}')
                    result_processing_end = time.time()
                    result_processing_duration = log_timing(f"Result Processing (Query {query_id})", result_processing_start, result_processing_end)
                            
                    print("final ranklist: ",ranklist)
                    query_end_time = time.time()
                    query_total_time = query_end_time - query_start_time
                    spend_time.append(query_total_time)
                    print(f"\n🏁 Total time for query {query_id}: {query_total_time:.2f} seconds")
                    print("=" * 80)
                    # === Evaluation ===
                    eval_result = save_eval_result(database_name,search_model_name,query_id,gt,fp,unjudge_dict,gt_file,ranklist,eval_k,eval_file,temp_file=savefile.split('.')[0]+'_temp_eval.txt')
                    eval_results[query_id] = eval_result
                    save_ranklist(query_id,ranklist_str,savefile)
                    query_tqdm.update(1)
                except Exception as e:
                    print(f"Error processing query {query_id}: {e}")
                    continue
            else:
                # === Reading Existing Results ===
                file_read_start = time.time()
                with open(savefile, 'r') as reader:
                    ranklist_str = reader.read()
                ranklist = ranklist_str.split()
                file_read_end = time.time()
                log_timing(f"File Read (Query {query_id})", file_read_start, file_read_end)
                # === Evaluation ===
                eval_result = save_eval_result(database_name,search_model_name,query_id,gt,fp,unjudge_dict,gt_file,ranklist,eval_k,eval_file)
                eval_results[query_id] = eval_result
    finally:
        if search_model_name == 'IITV':
            # === Server Cleanup (at the end of all queries) ===
            stop_servers(server_processes)
        else:
            pass


if __name__ == '__main__':
    args = parse_args()
    main(args.database_name,args.MLLM_model_id,args.search_model_name,args.featurename,args.rootpath,args.eval_k,args.examine_number,args.llama_device,args.vlm_device,args.search_device,args.server_device,args.BLIP2_server,args.ImageBind_server,args.BLIP2_feature_file,args.ImageBind_feature_file,args.MAX_ITER,args.greedy_action,args.start_query_idx,args.action_type,args.reformulation_type)