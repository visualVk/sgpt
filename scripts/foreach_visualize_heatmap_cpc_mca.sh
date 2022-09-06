#!/bin/bash
img_ids=(37 47 36 6638 5978 3905 8506 65 67 89)

for img_id in ${img_ids[*]}
do
    python visualize_heatmap.py \
        --task cpc_mca \
        --cam_dir outputs/cpc_mca \
        --pretrained logs/detr-r100-cpc-mca-hico-115-3/checkpoint0080.pth \
        --image_id $img_id \
        --device cuda:3
    
    echo "====================>>>>>finish $img_id<<<<<===================="
done