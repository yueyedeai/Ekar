#!/usr/bin/env bash

data_dir="data/movie-one"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
interact_relation="film.user.interacted_film"
entity_dim=32
relation_dim=32
emb_2D_d1=4
emb_2D_d2=8
num_epochs=1000
batch_size=512
train_batch_size=512
dev_batch_size=128
learning_rate=0.01
grad_norm=5
beam_size=64
emb_dropout_rate=0.1
feat_dropout_rate=0.1
hidden_dropout_rate=0.1


num_negative_samples=100 # useless?
margin=0.5 # useless?

