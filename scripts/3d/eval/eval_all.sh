MODEL=$1
FRAMES=$2

set -x

bash scripts/3d/eval/eval_sqa3d.sh $MODEL uniform $FRAMES
bash scripts/3d/eval/eval_scanqa.sh $MODEL uniform $FRAMES
bash scripts/3d/eval/eval_scan2cap.sh $MODEL uniform $FRAMES
bash scripts/3d/eval/eval_scanrefer.sh $MODEL uniform $FRAMES
bash scripts/3d/eval/eval_multi3drefer.sh $MODEL uniform $FRAMES
