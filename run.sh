#!/usr/bin/env bash
dataset=$1
model=$2
gpuID=$3
args=${@:4}
if [ $model == 'process' ]
then
    ./experiment.sh configs/${dataset}.sh --process_data ${gpuID} $args
elif [ $model == 'conve' ]
then
    ./experiment-conve.sh configs/${dataset}-conve.sh --train ${gpuID} \
    --train_raw_graph $args
elif [ $model == 'complex' ]
then
    ./experiment-emb.sh configs/${dataset}-complex.sh --train ${gpuID} \
    --train_raw_graph $args
elif [ $model == 'distmult' ]
then
    ./experiment-emb.sh configs/${dataset}-distmult.sh --train ${gpuID} \
    --train_raw_graph $args
elif [ $model == 'rl' ]
then
    ./experiment.sh configs/${dataset}.sh --train ${gpuID} --run_analysis $args
elif [ $model == 'rl.rs' ]
then
    ./experiment-rs.sh configs/${dataset}-rs.sh --train ${gpuID} --run_analysis \
    --use_pretrain --fix_embedding $args
fi




