CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=3 \
        --master_port=32002 \
        --use_env \
        main.py \
        --pretrained logs/detr-r50-cpc-mca-vcoco-115/checkpoint0073.pth \
        --hoi \
        --epochs 90 \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --eval