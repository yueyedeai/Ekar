#!/usr/bin/env bash
echo "run convE model"
bash run.sh movie-one conve $1 --tag tmp --num_epochs 1
echo "run Ekar"
bash run.sh movie-one rl.rs $1 --tag tmp --num_epochs 1
