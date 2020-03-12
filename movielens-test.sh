#!/usr/bin/env bash
bash run.sh movie-one rl.rs $1 --test_metrics --reward_as_score --stdout --show_recommend
# --test_metrics指定测试模型
# --reward_as_score指定模型最后的打分方式
# --stdout将结果输出在屏幕上
# --show_recommend显示每个用户推荐的商品