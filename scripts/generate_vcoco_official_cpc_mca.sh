python generate_vcoco_official.py \
        --backbone resnet50 \
        --param_path logs/detr-r50-cpc-mca-vcoco-115-bs4-jsd/checkpoint0078.pth \
        --save_path vcoco_jsd_bs4_r50_78.pickle \
        --hoi_path data/v-coco