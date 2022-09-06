CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=3 \
        --use_env \
        --master_port 32002 \
        main.py \
        --pretrained logs/detr-r100-cpc-mca-hico-115-3/checkpoint0080.pth \
        --batch_size 4 \
        --hoi \
        --save_hoi_path data/hico_20160224_det/add_file/hico_best_cpc_mca_bs4_3.json \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet101 \
        --eval