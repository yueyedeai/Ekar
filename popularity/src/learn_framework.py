"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Base learning framework.
"""

import os
import sys
import random
import shutil
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops
from time import time, sleep
from collections import defaultdict
from IPython import embed

class LFramework(nn.Module):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None

        self.inference = not args.train
        self.run_analysis = args.run_analysis
        self.case_study = args.case_study
        self.max_decrease_count = args.max_decrease_count

        self.kg = kg
        self.mdl = mdl
        self.K = args.K
        print('{} module created'.format(self.model))

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def run_train(self, train_data, dev_data):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []
        # last_loss = 100.0

        t0 = time()
        t1 = t0
        t_train = 0
        t_valid = 0
        n_valid = 0

        t_loss = 0
        t_loss_0 = 0
        t_back = 0
        t_back_0 = 0
        t_step = 0
        t_step_0 = 0
        decrease_count = 0

        for epoch_id in range(self.start_epoch, self.num_epochs):
            sys.stdout.flush()
            print('Epoch {}'.format(epoch_id))
            if epoch_id == self.start_epoch and self.rl_variation_tag.startswith('rs'):
                # Reward shaping module sanity check:
                #   Make sure the reward shaping module output value is in the correct range
                train_scores = self.test_fn(train_data)
                # dev_scores = self.test_fn(dev_data)
                print('Train set average fact score: {}'.format(float(train_scores.mean())))
                # print('Dev set average fact score: {}'.format(float(dev_scores.mean())))

            # Update model parameters
            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            entropies = []
            if self.run_analysis:
                rewards = None
                fns = None
            t_loss = 0
            t_back = 0
            t_step = 0
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):
                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue

                t_loss_0 = time()
                loss = self.loss(mini_batch)
                t_loss += time() - t_loss_0

                t_back_0 = time()
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)
                t_back += time() - t_back_0

                t_step_0 = time()
                self.optim.step()
                t_step += time() - t_step_0

                batch_losses.append(loss['print_loss'])
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                if self.run_analysis:
                    if rewards is None:
                        rewards = loss['reward']
                    else:
                        rewards = torch.cat([rewards, loss['reward']])
                    if fns is None:# what's this? --dzj
                        fns = loss['fn']
                    else:
                        fns = torch.cat([fns, loss['fn']])

            print ("time for calculate loss = %.2f" % t_loss)
            print ("time for backward = %.2f"       % t_back)
            print ("time for step = %.2f"           % t_step)

            _t_train = time() - t1
            print ("training time for epoch %d = %.1f" % (epoch_id, _t_train))
            sys.stdout.flush()
            t_train += _t_train

            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))

            # epoch_loss = np.mean(batch_losses)
            # if (epoch_loss > last_loss):
            #     self.learning_rate *= 0.5
            #     print("learning rate decay to %f" % self.learning_rate)
            #     if self.learning_rate < 1e-5:
            #         self.num_epochs = epoch_id + 1
            #         break
            #     for param_group in self.optim.param_groups:
            #         param_group['lr'] = self.learning_rate
            # last_loss = epoch_loss

            if entropies:
                stdout_msg += '\nentropy = {}'.format(np.mean(entropies))
            print(stdout_msg)
            # self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                hit_ratio = num_hits / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                num_fns = float(fns.sum())
                fn_ratio = num_fns / len(fns)
                print('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            if epoch_id % self.num_peek_epochs == 0:
                n_valid += 1
                _t_valid = time()
                self.eval()
                self.batch_size = self.dev_batch_size
                dev_scores = self.forward(dev_data, verbose=False)
                print('Dev set performance: (include test set labels)')
                NDCG, Precison, Recall = src.eval.NDCG_Precision_Recall(dev_data, dev_scores, self.kg.all_objects, self.kg.item_set, K=self.K, verbose=True)
                metrics = sum(NDCG) + sum(Precison) + sum(Recall)
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}\n'.format(epoch_id))
                        for i in range(len(self.K)):
                            o_f.write("NDCG@%d = %.4f\n" % (self.K[i], NDCG[i]))
                        for i in range(len(self.K)):
                            o_f.write("Precion@%d = %.4f\n" % (self.K[i], Precison[i]))
                        for i in range(len(self.K)):
                            o_f.write("Recall@%d = %.4f\n" % (self.K[i], Recall[i]))
                        # We can add more information

                else:
                    # Early stopping
                    if n_valid > self.max_decrease_count:
                        if metrics < dev_metrics_history[-1]:
                            decrease_count += 1
                        else:
                            decrease_count = 0
                        if decrease_count >= self.max_decrease_count:
                            self.num_epochs = epoch_id + 1
                            break

                dev_metrics_history.append(metrics)
                if self.run_analysis:
                    num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                    dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                    hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                    fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                    if epoch_id == 0:
                        with open(num_path_types_file, 'w') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'w') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                    else:
                        with open(num_path_types_file, 'a') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'a') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                t_valid += time() - _t_valid

            print ("time for epoch %d = %.1f" % (epoch_id, time() - t1))
            sys.stdout.flush()
            t1 = time()
        if self.num_epochs - self.start_epoch > 0:
            print ("*************************")
            print ("*** Average time for training = %.1f" % (t_train / (self.num_epochs - self.start_epoch)))
            print ("*** Average time for evaluate = %.1f" % (t_valid / n_valid))
            print ("*** Average time for each epoch = %.1f" % ((time() - t0) / (self.num_epochs - self.start_epoch)))

    def run_test_metrics(self, test_data):
        self.print_all_model_parameters()
        print ("size of item set = ", len(self.kg.item_set))
        t0 = time()
        self.load_checkpoint(os.path.join(self.model_dir, 'model_best.tar'))

        K = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        cnt = np.zeros(len(K))
        for i in range(self.n_user):
            pu = self.user_p_embeddings.weight[i]
            pu = torch.sigmoid(pu).item()
            for j in range(len(K)):
                if pu > K[j]:
                    cnt[j] += 1
        for j in range(len(K)):
            print ("> %f : %d (%f)%%" % (K[j], cnt[j], 100 * cnt[j] / self.n_user))
        sys.stdout.flush()
        self.eval()
        self.batch_size = self.dev_batch_size
        test_scores = self.forward(test_data)

        NDCG, Precison, Recall = src.eval.NDCG_Precision_Recall(test_data, test_scores, self.kg.all_objects, self.kg.item_set, K=self.args.K, verbose=True)
        metrics = dict()
        metrics['K']    = self.K
        metrics['NDCG'] = NDCG
        metrics['Precison']    = Precison
        metrics['Recall']      = Recall
        print ("total test time = %.4f\n" % (time() - t0))


        return metrics

    def run_case_study(self, test_data):
        self.print_all_model_parameters()
        print ("size of item set = ", len(self.kg.item_set))
        self.load_checkpoint(os.path.join(self.model_dir, 'model_best.tar'))

        self.eval()
        self.batch_size = self.dev_batch_size
        test_scores = self.forward_case_study(test_data)

    def forward_case_study(self, examples, verbose=False):
        pred_scores = []
        all_meta_path_dict = defaultdict(int)
        all_meta_path_sum  = 0
        pos_meta_path_dict = defaultdict(int)
        pos_meta_path_sum  = 0
        show_case_f = open(os.path.join(self.args.model_dir, "show_case.txt"), "w")
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            pred_score, _all_meta_path_dict, _pos_meta_path_dict = \
                self.predict(mini_batch, verbose=verbose, case_study=True, show_case=True, show_case_f=show_case_f)
            pred_scores.append(pred_score[:mini_batch_size])
            for k, v in _all_meta_path_dict.items():
                all_meta_path_dict[k] += v
                all_meta_path_sum += v
            for k, v in _pos_meta_path_dict.items():
                pos_meta_path_dict[k] += v
                pos_meta_path_sum += v

        show_case_f.close()
        print("********All meta path**********")
        for k, v in all_meta_path_dict.items():
            print ("%s:%d(%.1f)%%" % (k, v, 100.0 * v / all_meta_path_sum))
        print("********Positive meta path**********")
        for k, v in pos_meta_path_dict.items():
            print ("%s:%d(%.1f)%%" % (k, v, 100.0 * v / pos_meta_path_sum))
        scores = torch.cat(pred_scores)
        return scores

    def forward(self, examples, verbose=False):
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            pred_score = self.predict(mini_batch, verbose=verbose)
            pred_scores.append(pred_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        return scores

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        # out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        # if is_best:
        #     best_path = os.path.join(self.model_dir, 'model_best.tar')
        #     shutil.copyfile(out_tar, best_path)
        #     print('=> best model updated \'{}\''.format(best_path))
        # else:
        #     torch.save(checkpoint_dict, out_tar)
        #     print('=> saving checkpoint to \'{}\''.format(out_tar))

        if is_best:
            best_path = os.path.join(self.model_dir, 'model_best.tar')
            torch.save(checkpoint_dict, best_path)

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file)
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            print(r, '{}'.format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
