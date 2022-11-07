#!/bin/bash -l
#
#SBATCH --job-name=test-dino-train
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/logs/%x-%j-on-%N.out
#SBATCH -e /home/%u/logs/%x-%j-on-%N.err

echo "Job is running on" ${hostname}

#activate conda environment
conda activate dino

#copy code to local machine
mkdir -p /scratch/zinnen
cp -r /net/cluster/zinnen/detectors/ODOR-DINO/ /scratch/zinnen/

#untar data to local machine
WORKDIR="/scratch/zinnen/ODOR-DINO"

pwd

cd ${WORKDIR}
tar -xf /net/cluster/shared_dataset/ODOR/ODOR_coco.tar 

#start training
python main.py\
	--output_dir /net/cluster/zinnen/logs/DINO/lme/\
	-c ${WORKDIR}/config/DINO/ODOR_test.py\
	--coco_path ${WORKDIR}/coco-style/ \
	--pretrain_model_path /net/cluster/zinnen/models/swin_large_patch4_window12_384_22k.pth\
	--finetune_ignore label_enc.weight class_embed\
	--save_results\
	--save_log\
	--no_distribute\
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=/net/cluster/zinnen/models
	
echo 'yay'
