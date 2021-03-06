#!/usr/bin/env bash

data_dir="data/movie-one"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="False"
interact_relation="film.user.interacted_film"

bandwidth=256
entity_dim=32
relation_dim=32
history_dim=32
emb_2D_d1=4
emb_2D_d2=8
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
num_epochs=30
num_peek_epochs=1
batch_size=512
train_batch_size=512
dev_batch_size=4
learning_rate=0.005
grad_norm=5
emb_dropout_rate=0.0
ff_dropout_rate=0.0
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.05
entity_history="False"
history_only="True"
beam_size=64

distmult_state_dict_path="model/"
complex_state_dict_path="model/"
conve_state_dict_path="model/movie-one-conve-RV-RG-xavier-32-32-0.01-32-3-0.1-0.1-0.1-0.1/model_best.tar"
