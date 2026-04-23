import os
import numpy as np
from utils import readQuerySet


# database_names = ['iacc.3','v3c1','v3c2']
database_names = ['v3c1']
for database_name in database_names:
    if database_name == 'iacc.3':
        query_sets = ['tv16','tv17','tv18']
    elif database_name == 'v3c1':
        query_sets = ['tv19','tv20','tv21']
    elif database_name == 'v3c2':
        query_sets = ['tv22','tv23','tv24']

    search_model = "IITV"
    resultpath = f"/mnt_nas1/nikki/data/VBS_data/dataset/TRECVid/{database_name}/results/ViSA_zero_shot__MLLM_Qwen3-VL-8B-Instruct_search_model_{search_model}_eval_k_1000_examine_number_50_MAX_ITER_60_action_type_reasoning_reformulation_type_with_action_reasoning"
    result_csv = os.path.join(f"/mnt_nas1/nikki/data/VBS_data/dataset/TRECVid/{database_name}/results",f'eval_results_{database_name}_{search_model}_1000.csv')

    with open(result_csv, 'w') as f:
        f.write('search_model,query_id,precision,recall,map,xinfAP,match_num,unmatch_num,unjudge_num,k\n')
                
    queryset2query_ids,queryset2gt = readQuerySet(database_name)
    for query_set in query_sets:
        query_ids = queryset2query_ids[query_set]
        precisions = []
        recalls = []
        maps = []
        xinfAPs = []        
        match_nums = []
        unmatch_nums = []
        unjudge_nums = []
        ks = []
    
        for query_id in query_ids:
            resultfile = os.path.join(resultpath,f'eval_results_{query_id}.txt')
            precision = None
            recall = None
            map = None
            xinfAP = None
            match_num = None
            unmatch_num = None
            unjudge_num = None
            k = None
            if os.path.exists(resultfile):
                """ file content:
                query_id: 611
                precision: 0.82
                recall: 0.20654911838790932
                map: 0.82
                match_num: 82
                unmatch_num: 2
                unjudge_num: 10
                k: 100

                """
                with open(resultfile, 'r') as f:
                    result = f.read()
                    result = result.split('\n')
                    for line in result:
                        if line.startswith('query_id:'):
                            query_id = line.split(':')[1].strip()
                        elif line.startswith('precision:'):
                            precision = line.split(':')[1].strip()
                        elif line.startswith('recall:'):
                            recall = line.split(':')[1].strip()
                        elif line.startswith('map:'):
                            map = line.split(':')[1].strip()
                        elif line.startswith('xinfAP:'):
                            xinfAP = line.split(':')[1].strip()
                        elif line.startswith('match_num:'):
                            match_num = line.split(':')[1].strip()
                        elif line.startswith('unmatch_num:'):
                            unmatch_num = line.split(':')[1].strip()
                        elif line.startswith('unjudge_num:'):
                            unjudge_num = line.split(':')[1].strip()
                        elif line.startswith('k:'):
                            k = line.split(':')[1].strip()     
                with open(result_csv, 'a') as f:
                    f.write(f'{search_model},{query_id},{precision},{recall},{map},{xinfAP},{match_num},{unmatch_num},{unjudge_num},{k}\n')
                # print(f'eval_result: {query_id,precision,recall,map,xinfAP,match_num,unmatch_num,unjudge_num,k}')
                precisions.append(float(precision))
                recalls.append(float(recall))
                maps.append(float(map))
                xinfAPs.append(float(xinfAP))
                match_nums.append(int(match_num))
                unmatch_nums.append(int(unmatch_num))
                unjudge_nums.append(int(unjudge_num))
                ks.append(int(k))
            else:
                print(f'resultfile {resultfile} not found')
                with open(result_csv, 'a') as f:
                    f.write(f'{query_id},None,None,None,None,None,None,None\n')
        ##print empty line between each query_set
        with open(result_csv, 'a') as f:
            f.write('\n')
        print(f'query_set: {query_set}')
        print(f'precision: {np.mean(precisions)}')
        print(f'recall: {np.mean(recalls)}')
        print(f'map: {np.mean(maps)}')
        print(f'xinfAP: {np.mean(xinfAPs)}')
        print(f'match_num: {np.mean(match_nums)}')
        print(f'unmatch_num: {np.mean(unmatch_nums)}')
        print(f'unjudge_num: {np.mean(unjudge_nums)}')
        print(f'k: {np.mean(ks)}')
    print(f'eval result has been saved to {result_csv}')