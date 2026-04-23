import os
from utils import get_database_name
from eval.readGTandPrint import readGT
import numpy as np
from tqdm import tqdm

def parse_result(res):
    resp = {}
    lines = res.split('\n')
    for line in lines:
        elems = line.split()
        if 'infAP' in elems:
            line_fileter = line.split('\t\t')
        if 'infAP' == elems[0] and 'all' in line:
            return float(elems[-1])
        
def TRECVid_AVS_eval(query_id,gt_file,ranklist,topk=1000,temp_file='temp_eval.txt'):
    ##creat temp file with ranklist formating as 1641 0 shot07070_12 1 9999 VIREO
    lines = [f'1{query_id} 0 {item} {item_idx+1} {999-item_idx} VIREO' for item_idx,item in enumerate(ranklist)]
    with open(temp_file, 'w') as writer:
        writer.writelines('\n'.join(lines))
    cmd = 'perl sample_eval.pl -q %s %s %d' % (gt_file, temp_file, topk)
    # print(cmd)
    res = os.popen(cmd).read()
    
    resp = parse_result(res)
    ##try to remove the temp file
    try:
        os.remove(temp_file)
    except:
        pass
    return resp
    
def main():
    
    rootpath = "/data/nikki/mount/"
    query_sets = ['tv19','tv20','tv21','tv22','tv23']
    eval_k = 1000

    modelname  = "IITV"
    for query_set in query_sets:
        database_name = get_database_name(query_set)
        query_file = os.path.join(rootpath,database_name,'TextData',query_set+'.avs.txt')

        if not os.path.exists(query_file):
            print(f"query_file not found: {query_file}")
            continue
        with open(query_file, 'r') as reader:
            lines = reader.readlines()
            
        ##get the queries
        query_ids,query_contents = [],[]
        for line in lines:
            query_id,query_content = line.split(' ',1)
            query_ids.append(query_id)
            query_contents.append(query_content)
        ##get the ranklist
        query_set_path = f"/home/nikki/code/active_learning_VS/Improved_ITV_results/{query_set}.avs.txt/id.sent.sim.0.99.combinedDecodedConcept_theta0_5_score"
        eval_savepath = '/'.join(query_set_path.split('/')[:-1])
        map_results = []
        for query_id,query_content in zip(query_ids,query_contents):
            gt_file = os.path.join(rootpath,database_name,'TextData',f'avs.qrels.{query_set}.{"1"+query_id}')

            rank_list_file  = query_set_path+'.'+query_id
            with open(rank_list_file, 'r') as reader:
                ranklist = reader.readlines()
            ranklist =[item.split(' ')[0] for item in ranklist]
            topk = 1000
            map = TRECVid_AVS_eval(query_id,gt_file,ranklist,topk)
            map_results.append(map)
            print(f"query_id: {query_id}, MAP: {map:.4f}")
        map_results = np.array(map_results)
        print(f"query_set: {query_set}, MAP: {np.mean(map_results):.4f}")

if __name__ == '__main__':
    main()