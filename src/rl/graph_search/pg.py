"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient (REINFORCE algorithm) training and inference.
"""

import torch
import numpy as np
import sys
from src.learn_framework import LFramework
import src.rl.graph_search.beam_search as search
import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
from collections import defaultdict
from time import time
from IPython import embed


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        # self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval

        # Inference hyperparameters
        self.beam_size = args.beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

        self.bias = torch.zeros(self.kg.num_entities) - 1
        self.bias[list(self.kg.item_set)] = 0
        self.bias = self.bias.cuda()
        self.rollout_inference = args.rollout_inference

        self.args = args

    def get_reward_matrix(self, train_data, n_entity):
        self.id2uid = torch.zeros(n_entity, dtype=torch.int64)
        n_user = 0
        for triple in train_data:
            e1, e2, r = triple
            if self.id2uid[e1] == 0:
                n_user += 1
                self.id2uid[e1] = n_user
        self.reward_matrix = torch.zeros([n_user + 1, n_entity], dtype=torch.uint8)
        for triple in train_data:
            e1, e2, r = triple
            self.reward_matrix[self.id2uid[e1]][e2] = 1
        self.id2uid = self.id2uid.cuda()
        self.reward_matrix = self.reward_matrix.cuda()
        # embed()

    def reward_fun(self, e1, r, e2, pred_e2):
        # sys.stderr.write("******" + str(type(pred_e2)))
        # sys.stderr.write("******" + str(type(e2)))
        bias = self.bias[pred_e2]
        if self.args.reward_matrix:
            return  self.reward_matrix[self.id2uid[e1], pred_e2].float()
        else:
            return (pred_e2 == e2).float() + bias

    def loss(self, mini_batch):
        # t0 = time()

        # def stablize_reward(r):
        #     # We don't use any baseline function yet.
        #     r_2D = r.view(-1, self.num_rollouts)
        #     if self.baseline == 'avg_reward':
        #         stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
        #     elif self.baseline == 'avg_reward_normalized':
        #         stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
        #     else:
        #         raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
        #     stabled_r = stabled_r_2D.view(-1)
        #     return stabled_r
    
        e1, e2, r = self.format_batch(mini_batch, num_labels = self.kg.num_entities, num_tiles=self.num_rollouts)

        # t_rollout = time()
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)
        # t_rollout = time() - t_rollout

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        # if self.baseline != 'n/a':
        #     final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        # print ("time for rollout in this batch = %.2f" % (t_rollout))
        # print ("time for loss in this batch = %.2f"    % (time() - t0))
        sys.stdout.flush()
        return loss_dict

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.???
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        # t0 = time()

        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]

            # t_transit_0 = time()
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            # t_transit += time() - t_transit_0

            # t_sample_0 = time()
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']    #(next_r, next_e)
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)
            # t_sample += time() - t_sample_0

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        # self.record_path_trace(path_trace)

        # print ("time for transit = %.2f" % t_transit)
        # print ("time for sample = %.2f"  % t_sample)
        # print ("time for rollout = %.2f" % (time() - t0))
        # sys.stdout.flush()

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }


    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(self, mini_batch, verbose=False, case_study=False, show_case=False, show_case_f=None):

        def get_relation(relation_id):
            if relation_id == self.kg.self_edge:
                return '<null>'
            else:
                return self.kg.id2relation[relation_id]

        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch, num_labels=kg.num_entities)
        if self.rollout_inference:
            _e1 = e1.unsqueeze(1).repeat((1, self.beam_size)).view(-1)
            _r = r.unsqueeze(1).repeat((1, self.beam_size)).view(-1)
            _e2 = torch.zeros(_e1.shape).unsqueeze(1)
            self.action_dropout_rate = 0
            output = self.rollout(_e1, _r, _e2, num_steps=self.num_rollout_steps)
            _pred_e2 = output['pred_e2']
            _scores = self.reward_fun(_e1, _r, _e1, _pred_e2)
            pred_e2s = _pred_e2.view(e1.shape[0], -1)
            pred_e2_scores = _scores.reshape(pred_e2s.shape[0], -1)
        else:
            beam_search_output = search.beam_search(
                pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size,
                return_search_traces=case_study,
                use_action_space_bucketing=self.args.use_action_space_bucketing,
                multi_path=self.args.multi_path)
            pred_e2s = beam_search_output['pred_e2s']
            if self.reward_as_score:
                _e1 = e1.unsqueeze(1).repeat((1, pred_e2s.shape[1])).view(-1)
                _r = r.unsqueeze(1).repeat((1, pred_e2s.shape[1])).view(-1)
                _pred_e2 = pred_e2s.view(-1)
                _scores = self.reward_fun(_e1, _r, None, _pred_e2)
                pred_e2_scores = _scores.reshape(pred_e2s.shape[0], -1)
            else:
                pred_e2_scores = beam_search_output['pred_e2_scores']

        if case_study:
            all_meta_path_dict = defaultdict(int)
            pos_meta_path_dict = defaultdict(int)
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    meta_path = ''
                    for k in range(len(search_traces)):
                        relation_id = int(search_traces[k][0][ind])
                        entity_id   = int(search_traces[k][1][ind])
                        search_trace.append((relation_id, entity_id))
                        if (k > 0):
                            relation = get_relation(relation_id)
                            meta_path += '==>' + relation
                    score = float(pred_e2_scores[i][j])
                    path  = ops.format_path(search_trace, kg)
                    if verbose:
                        print('beam {}: score = {} \n<PATH> {}'.format(
                        j, score, path))
                    all_meta_path_dict[meta_path] += 1
                    if (e2[i][pred_e2s[i][j]] != 0):
                        pos_meta_path_dict[meta_path] += 1
                        if (show_case and j <= 20):
                            show_case_f.write('beam {}: score = {} \n<PATH> {}\n'.format(
                                j, score, path))
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        if case_study == True:
            return pred_scores, all_meta_path_dict, pos_meta_path_dict
        return pred_scores

