# ODOR-DINO <img src="figs/dinosaur.png" width="30">

This is an adapted version of the official PyTorch [implementation](https://github.com/IDEA-Research/DINO) for DINO. It is used in the Odeuropa project to train our baseline models for the detection of olfactory objects.


## Installation

Please refer to the original repository installation [instructions](https://github.com/IDEA-Research/DINO#installation).


## Fine-Tune
Organize your data in coco format, i.e. 
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```
Training can be started using the following command
`python main.py -c config/DINO/ODOR_swin_50ep.py --pretrain_model_path ${PATH_TO_SWIN_MODEL} --finetune_ignore n label_enc.weight class_embed --options backbone_dir=${PATH_TO_DIR_CONTAINING_SWIN_MODEL} --coco_path ${PATH_TO_COCO}`

## Inference

Inference can be done using the `inference_odor.ipynb` script.
