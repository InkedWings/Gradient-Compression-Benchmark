#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A sbi-fair
#PBS -e /home/zye25/Gradient-Compression-Benchmark/logs/topk_0.8_exp.err
#PBS -o /home/zye25/Gradient-Compression-Benchmark/logs/topk_0.8_exp.out

PRELOAD="module load conda ; "
PRELOAD+="conda activate base ; "

WDIR="/home/zye25/Gradient-Compression-Benchmark/"

CMD="cd $WDIR ; "

CMD+="python main.py --model_name topk_0.8_with_ef --compression topk --compression_ratio 0.8 --memory residual ; "


FULL_CMD="$PRELOAD $CMD $@ "
echo "Training Command: $FULL_CMD "

eval $FULL_CMD