#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

add_reversed_training_edges_flag=''
if [[ $add_reversed_training_edges = *"True"* ]]; then
    add_reversed_training_edges_flag="--add_reversed_training_edges"
fi
group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi

cmd="python3 -u -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --interact_relation $interact_relation \
    --model $model \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --emb_2D_d1 $emb_2D_d1 \
    --emb_2D_d2 $emb_2D_d2 \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --learning_rate $learning_rate \
    --grad_norm $grad_norm \
    --beam_size $beam_size \
    --emb_dropout_rate $emb_dropout_rate \
    --hidden_dropout_rate $hidden_dropout_rate \
    --feat_dropout_rate $feat_dropout_rate \
    $group_examples_by_query_flag \
    $add_reversed_training_edges_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

$cmd
