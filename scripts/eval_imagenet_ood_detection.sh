EXP_NAME='exp1'
ID='ImageNet'
DATA_ROOT='../GL-MCM/datasets'


CKPT=ViT-B/16

python eval_id_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score GL-MCM --root-dir ${DATA_ROOT} --lambda_local 0.5
