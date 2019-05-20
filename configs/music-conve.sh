#!/usr/bin/env bash

data_dir="data/music"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
interact_relation="music.user.interacted_artist"
entity_dim=32
relation_dim=32
emb_2D_d1=4
emb_2D_d2=8
num_rollouts=1
num_epochs=1000
batch_size=1024
train_batch_size=1024
dev_batch_size=128
learning_rate=0.01
grad_norm=5
beam_size=64
emb_dropout_rate=0.2
feat_dropout_rate=0.1
hidden_dropout_rate=0.1
