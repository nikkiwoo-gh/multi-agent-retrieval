#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava


database_name="v3c1"
MLLM_model_id="/mnt_nas1/shared/Qwen3-VL-8B-Instruct/"
search_model_name="IITV"
featurename="Improved_ITV"
rootpath="/mnt_nas1/ABC/data/VBS_data/dataset/TRECVid/"
eval_k=1000
examine_number=50
vlm_device=cuda:1
search_device=cuda:1
server_device=cuda:1
BLIP2_server="./tmp/BLIP2_ViSA.sock"
ImageBind_server="./tmp/ImageBind_ViSA.sock"
BLIP2_feature_file="./tmp/BLIP2_ViSA_feature.npy"
ImageBind_feature_file="./tmp/ImageBind_ViSA_feature.npy"
MAX_ITER=60
action_type="reasoning"
reformulation_type="with_action_reasoning"
start_query_idx=0

python main.py --database_name=$database_name --MLLM_model_id=$MLLM_model_id --search_model_name=$search_model_name --featurename=$featurename --rootpath=$rootpath --eval_k=$eval_k --examine_number=$examine_number --vlm_device=$vlm_device --search_device=$search_device --server_device=$server_device --BLIP2_server=$BLIP2_server --ImageBind_server=$ImageBind_server --BLIP2_feature_file=$BLIP2_feature_file --ImageBind_feature_file=$ImageBind_feature_file --MAX_ITER=$MAX_ITER --action_type=$action_type --reformulation_type=$reformulation_type