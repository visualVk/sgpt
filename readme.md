# SGPT: The Secondary Path Guides the Primary Path in Transformers for HOI Detection

## Pre-request

- pretrained model: [qpic](https://github.com/hitachi-rd-cv/qpic)
- data file format followed by qpic
- hico dataset can be downloaded from [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
- vcoco dataset can be download from [here](https://github.com/s-gupta/v-coco)
- Please refer to qpic for data preparation from [here]([qpic](https://github.com/hitachi-rd-cv/qpic))

## Training

```shell
# vcoco r50
sh scripts/train_vcoco_50.sh
# hico r50
sh scripts/train_hico_50.sh
```

## Evalution

### vcoco
1. generate pickle file by offical code
2. evaluating by offical
```shell
sh scripts/generate_vcoco_official.sh
```

### hico
** two options: **
- result of evalution displayed in logs of training.
- evaluating by scripts offered by ours.
```shell
sh scripts/eval_hico_101.sh
```