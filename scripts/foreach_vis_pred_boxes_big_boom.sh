#!/bin/bash

# for img_id in ${img_ids[*]}
for ((i=1; i<=7; i++))
do
    python vis_video.py \
        --image_id $i \
        --json_file logs/video/bigboom_part3.json \
        --img_root data/videos/part3/frames \
        --save_root part3
        # --device cuda:3
    
    echo "====================>>>>>finish $img_id<<<<<===================="
done