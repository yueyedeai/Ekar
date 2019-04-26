#!/usr/bin/env bash

data_dir="data/dbbook2014-one-onlyUI"
model="point"
group_examples_by_query="False"
use_action_space_bucketing="True"
interact_relation="http://interacted"

bandwidth=256
entity_dim=32
relation_dim=32
history_dim=32
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
num_epochs=100
num_wait_epochs=100
num_peek_epochs=1
bucket_interval=5
batch_size=512
train_batch_size=512
dev_batch_size=1
learning_rate=0.01
baseline="n/a"
grad_norm=5
emb_dropout_rate=0.0
ff_dropout_rate=0.0
action_dropout_rate=0.0
action_dropout_anneal_interval=1000
beta=0.05
relation_only="False"
beam_size=256

num_paths_per_entity=-1
margin=-1