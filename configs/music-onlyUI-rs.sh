#!/usr/bin/env bash

data_dir="data/music-onlyUI"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="False"
interact_relation="music.user.interacted_artist"

bandwidth=256
entity_dim=32
relation_dim=32
history_dim=32
emb_2D_d1=4
emb_2D_d2=8
history_num_layers=3
num_rollouts=20 # 20
num_rollout_steps=3
num_epochs=300
num_wait_epochs=300
num_peek_epochs=1
bucket_interval=10
batch_size=1024
train_batch_size=1024
dev_batch_size=16
learning_rate=0.01
baseline="n/a"
grad_norm=5
emb_dropout_rate=0.0 # 0.1
ff_dropout_rate=0.0 # 0.1
action_dropout_rate=0.5 # 0.1
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.05
relation_only="False"
entity_only="True"
beam_size=256

distmult_state_dict_path="model/"
complex_state_dict_path="model/"
conve_state_dict_path="model/music-onlyUI-conve-RV-xavier-32-32-0.01-32-3-0.0-0.1-0.1-0.1/model_best.tar"
#conve_state_dict_path="model/music-conve-RV-xavier-32-32-0.01-32-3-0.3-0.3-0.2-0.1/model_best.tar"
num_paths_per_entity=-1
margin=-1
