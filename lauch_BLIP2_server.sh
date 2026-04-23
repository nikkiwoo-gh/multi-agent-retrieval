#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lavis

rootpath="/mnt_nas1/ABC/data/VBS_data/dataset/TRECVid/"
BLIP2_server="./tmp/BLIP2_ViSA.sock"
BLIP2_feature_file="./tmp/BLIP2_ViSA_feature.npy"
device="cuda:1"
##echo the command
echo "python3 BLIP2_text_encoder_server.py --device=$device --BLIP2_server=$BLIP2_server --BLIP2_feature_file=$BLIP2_feature_file"

##run the command
python3 BLIP2_text_encoder_server.py --device=$device --BLIP2_server=$BLIP2_server --BLIP2_feature_file=$BLIP2_feature_file
