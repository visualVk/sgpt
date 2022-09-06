#!/bin/bash
img_ids=(10009 10010 10011 10012 10013)

# for img_id in ${img_ids[*]}
for ((i=10001; i<=13090; i++))
do
    python vis_video.py \
        --image_id $i
        # --device cuda:3
    
    echo "====================>>>>>finish $img_id<<<<<===================="
done