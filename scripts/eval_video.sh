CUDA_VISIBLE_DEVICES=4,6,7 python -m torch.distributed.launch \
        --nproc_per_node=3 \
        --use_env \
        video_read.py \
        --pretrained logs/detr-r100-cpc-mca-vcoco-115-bs4/checkpoint0080.pth \
        --output_dir logs/video \
        --hoi \
        --dataset_file video \
        --hoi_path data/videos/part3 \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --save_hoi_path logs/video/bigboom_part3.json \
        --backbone resnet101