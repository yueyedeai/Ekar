#!/usr/bin/env bash

data_dir="data/music"
model="point"
group_examples_by_query="False"

bandwidth=256
entity_dim=32
relation_dim=32
history_dim=32
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
num_epochs=100
num_peek_epochs=1
batch_size=1024
train_batch_size=1024
dev_batch_size=1
learning_rate=0.01
grad_norm=5
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.3
action_dropout_anneal_interval=1000
beta=0.05
relation_only="False"
beam_size=64
