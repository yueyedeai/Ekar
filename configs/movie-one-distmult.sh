#!/usr/bin/env bash

data_dir="data/movie-one"
model="distmult"
add_reversed_training_edges="True"
group_examples_by_query="True"
interact_relation="film.user.interacted_film"
entity_dim=32
relation_dim=32
num_epochs=1000
batch_size=512
train_batch_size=512
dev_batch_size=64
learning_rate=0.001
grad_norm=5
emb_dropout_rate=0.0
beam_size=64
