# lr_drop 40
CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=3 \
        --master_port=32001 \
        --use_env \
        main.py \
        --pretrained params/detr-r100-pre-hico.pth \
        --output_dir logs/detr-r100-cpc-mca-hico-115-3 \
        --batch_size 4 \
        --epochs 90 \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet101 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --sub_loss_aug_coef 1 \
        --obj_loss_aug_coef 1 \
        --verb_loss_aug_coef 0.5
