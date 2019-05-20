"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Compute Evaluation Metrics.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/evaluation.py
"""

import numpy as np
import pickle

import torch

from src.parse_args import args
from src.data_utils import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID
from src.utils.ndcg_p_r import eval_NDCG_P_R

def NDCG_Precision_Recall(examples, scores, all_answers, item_set, verbose=False,
                          K=np.array([10]), phase="test"):
    """
    Compute ranking based metrics.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    average_arrive = 0
    average_answer = 0
    for i, example in enumerate(examples):
        e1, e2, r = example
        # keep the item scores only
        item_list = list(item_set)
        item_score = scores[i, item_list]
        scores[i] = 0
        scores[i, item_list] = item_score

        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = scores[i, e2]
        average_arrive += torch.nonzero(target_score).shape[0]
        average_answer += target_score.shape[0]
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    average_arrive /= len(examples)
    average_answer /= len(examples)

    Precision = np.zeros(len(K), dtype=float)
    NDCG      = np.zeros(len(K), dtype=float)
    Recall    = np.zeros(len(K), dtype=float)
    for i, example in enumerate(examples):
        e1, e2, r = example
        label = np.zeros(scores.shape[1])
        label[e2] = 1
        _NDCG, _Precision, _Recall = eval_NDCG_P_R(scores[i].cpu().numpy(), label, K)
        NDCG      = NDCG + _NDCG
        Precision = Precision + _Precision
        Recall    = Recall + _Recall

    NDCG /= len(examples)
    Precision /= len(examples)
    Recall    /= len(examples)

    if verbose:
        for k in range(len(K)):
            print ('NDCG@%d = %.4f' % (K[k], NDCG[k]))
        for k in range(len(K)):
            print ('Precision@%d = %.4f' % (K[k], Precision[k]))
        for k in range(len(K)):
            print ('Recall@%d = %.4f' % (K[k], Recall[k]))
        print ("Average Arrive cnt = %.1f" % average_arrive)
        print ("Average Answer cnt = %.1f" % average_answer)

    return NDCG, Precision, Recall
