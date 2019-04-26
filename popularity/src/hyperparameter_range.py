"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Hyperparameter range specification.
"""

hp_range = {
    "emb_dropout_rate": [0, 0.2, 0.4],
    "feat_dropout_rate": [0, 0.2, 0.4],
    "hidden_dropout_rate": [0, 0.2, 0.4],
    "action_dropout_rate": [0.6, 0.8, 0.9],
    "ff_dropout_rate": [0, 0.2, 0.5],
    "learning_rate": [0.01, 0.001],
    "num_rollout_steps": [3, 5]
}
