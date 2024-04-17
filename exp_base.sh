#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A sbi-fair
#PBS -e /home/zye25/Gradient-Compression-Benchmark/logs/self_topk_0.25_ef.err
#PBS -o /home/zye25/Gradient-Compression-Benchmark/logs/self_topk_0.25_ef.out

PRELOAD="module load conda ; "
PRELOAD+="conda activate base ; "

WDIR="/home/zye25/Gradient-Compression-Benchmark/"

CMD="cd $WDIR ; "

CMD+="python main.py --model_name self_topk_0.25_ef --compression topk --compression_ratio 0.25 --memory residual"


FULL_CMD="$PRELOAD $CMD $@ "
echo "Training Command: $FULL_CMD "

eval $FULL_CMD