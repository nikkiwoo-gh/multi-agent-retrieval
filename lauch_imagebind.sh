#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate imagebind

rootpath="/mnt_nas1/ABC/data/VBS_data/dataset/TRECVid/"
ImageBind_server="./tmp/ImageBind_ViSA.sock"
ImageBind_feature_file="./tmp/ImageBind_ViSA_feature.npy"
device="cuda:1"

##echo the command
echo "python3 Imagebind_text_encoder_server.py --device=$device --ImageBind_server=$ImageBind_server --ImageBind_feature_file=$ImageBind_feature_file"

##run the command
python3 Imagebind_text_encoder_server.py --device=$device --ImageBind_server=$ImageBind_server --ImageBind_feature_file=$ImageBind_feature_file
