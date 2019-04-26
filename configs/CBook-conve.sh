#!/usr/bin/env bash

data_dir="data/CBook"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
interact_relation="book.user.interacted_book"
entity_dim=32
relation_dim=32
emb_2D_d1=4
emb_2D_d2=8
num_rollouts=1
bucket_interval=10
num_epochs=300 # !!
num_wait_epochs=50 # !!
batch_size=1024
train_batch_size=1024
dev_batch_size=128
learning_rate=0.001
grad_norm=5
emb_dropout_rate=0
hidden_dropout_rate=0
feat_dropout_rate=0
beam_size=128

num_negative_samples=100
margin=0.5
