EXP_NAME='exp1'
ID='COCO_single'
DATA_ROOT='./datasets'


CKPT=ViT-B/16

python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score MCM --root-dir ${DATA_ROOT} --num_ood_sumple 5000
python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score L-MCM --root-dir ${DATA_ROOT} --num_ood_sumple 5000
python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score GL-MCM --root-dir ${DATA_ROOT} --num_ood_sumple 5000

CKPT=RN50

python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score MCM --root-dir ${DATA_ROOT} --num_ood_sumple 5000
python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score L-MCM --root-dir ${DATA_ROOT} --num_ood_sumple 5000
python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score GL-MCM --root-dir ${DATA_ROOT} --num_ood_sumple 5000
