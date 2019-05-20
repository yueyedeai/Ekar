#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""

import copy
import itertools
import numpy as np
import os, sys
import random

import torch

from src.parse_args import parser
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.hyperparameter_range import hp_range
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ComplEx, ConvE, DistMult
from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.emb.emb import EmbeddingBasedMethod
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from src.utils.ops import flatten
from IPython import embed

torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # seed all GPUs

def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_enviroment(raw_kb_path, train_path, dev_path, test_path, args.add_reverse_relations)

def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    raw_graph_tag    = '-RG' if args.train_raw_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    if args.use_pretrain:
        initialization_tag += '-pretrain'
        if args.fix_embedding:
            initialization_tag += '-fix'

    # Hyperparameter signature
    if args.model.startswith('point'):
        if args.action_dropout_anneal_interval < 1000:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.num_rollout_steps,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.action_dropout_anneal_factor,
                args.action_dropout_anneal_interval,
                args.bandwidth,
                args.beta
            )
        else:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.num_rollout_steps,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta
            )
        if args.reward_shaping_threshold > 0:
            hyperparam_sig += '-{}'.format(args.reward_shaping_threshold)
    elif args.model == 'distmult':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model == 'complex':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model in ['conve', 'hypere', 'triplee']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.num_out_channels,
            args.kernel_size,
            args.emb_dropout_rate,
            args.hidden_dropout_rate,
            args.feat_dropout_rate,
            args.label_smoothing_epsilon
        )
    else:
        raise NotImplementedError

    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        raw_graph_tag,
        initialization_tag,
        hyperparam_sig
    )
    if args.entity_history:
        model_sub_dir += '-eh'
    elif args.history_only:
        model_sub_dir += '-ho'

    if random_seed:
        model_sub_dir += '.{}'.format(random_seed)
    if args.use_action_space_bucketing:
        model_sub_dir += '-bucket'

    if args.no_ground_truth_edge_mask:
        model_sub_dir += '-nogtm'

    if args.reward_matrix:
        model_sub_dir += '-rm'
    if args.remove_rs:
        model_sub_dir += '-rrs'
    if args.no_self_loop:
        model_sub_dir += '-noloop'

    if args.tag:
        model_sub_dir += '_' + args.tag
    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir

    if args.train and not args.test_metrics and not args.case_study:
        filename="output.txt"
        sys.stdout = open(os.path.join(args.model_dir, filename), "w")
        print (args)


def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)

    if args.model in ['point']:
        pn = GraphSearchPolicy(args)
        lf = PolicyGradient(args, kg, pn)
    elif args.model.startswith('point.rs'):
        pn = GraphSearchPolicy(args)
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'complex':
            fn = ComplEx(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'distmult':
            fn = DistMult(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
    elif args.model == 'complex':
        fn = ComplEx(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'distmult':
        fn = DistMult(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf

def test_metrics(lf):
    metrics = dict()
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')

    if args.test_train_data:
        print ("*********now test the training data")
        train_path = os.path.join(args.data_dir, 'train.triples')
        train_data = data_utils.load_triples(train_path, entity_index_path, relation_index_path,
                                           group_examples_by_query=True)
        metrics['train'] = lf.run_test_metrics(train_data)

    if args.test_dev_data:
        print ("*********now test the dev data")
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path,
                                           group_examples_by_query=True)
        metrics['dev'] = lf.run_test_metrics(dev_data)

    print("*********now test the test data")
    test_path = os.path.join(args.data_dir, 'test.triples')
    test_data = data_utils.load_triples(test_path, entity_index_path, relation_index_path,
                                       group_examples_by_query=True)
    metrics['test'] = lf.run_test_metrics(test_data)
    return metrics

def case_study(lf):
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')

    print("*********case study for the test data")
    test_path = os.path.join(args.data_dir, 'test.triples')
    test_data = data_utils.load_triples(test_path, entity_index_path, relation_index_path,
                                       group_examples_by_query=True)
    lf.run_case_study(test_data)

def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    raw_path = os.path.join(args.data_dir, 'raw.kb')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path,
                                       group_examples_by_query=True)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)

    if args.reward_matrix:
        entity2id, _ = data_utils.load_index(entity_index_path)
        n_entity = max([v for k, v in entity2id.items()]) + 1
        if args.model.startswith('point'):
            lf.get_reward_matrix(train_data, n_entity)
    lf.run_train(train_data, dev_data)

    test_metrics(lf)
    if args.model.startswith("point.rs"):
        lf.reward_as_score = True
        test_metrics(lf)

def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path

def run_experiment(args):

    if args.process_data:
        # Process knowledge graph data
        process_data()
    else:
        with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
            if args.grid_search:
                # Grid search
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.gs'.format(task, args.model)
                o_f = open(out_log, 'w')

                print("** Grid Search **")
                o_f.write("** Grid Search **\n")
                o_f.write(args.tune + '\n')
                hyperparameters = args.tune.split(',')

                if args.tune == '' or len(hyperparameters) < 1:
                    print("No hyperparameter specified.")
                    sys.exit(0)

                grid = hp_range[hyperparameters[0]]
                for hp in hyperparameters[1:]:
                    grid = itertools.product(grid, hp_range[hp])

                K    = {}
                NDCG = {}
                P    = {}
                R    = {}
                metrics_sum = {}
                grid = list(grid)
                print('* {} hyperparameter combinations to try'.format(len(grid)))
                o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
                o_f.close()

                for i, grid_entry in enumerate(list(grid)):

                    o_f = open(out_log, 'a')

                    if not (type(grid_entry) is list or type(grid_entry) is list):
                        grid_entry = [grid_entry]
                    grid_entry = flatten(grid_entry)
                    o_f.write('* Hyperparameter Set {}:\n'.format(i))
                    signature = ''
                    for j in range(len(grid_entry)):
                        hp = hyperparameters[j]
                        value = grid_entry[j]
                        if hp == 'bandwidth':
                            setattr(args, hp, int(value))
                        else:
                            setattr(args, hp, float(value))
                        signature += ':{}'.format(value)
                        o_f.write('* {}: {}\n'.format(hp, value))
                    initialize_model_directory(args)
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = test_metrics(lf)
                    K[signature]    = metrics['test']['K']
                    NDCG[signature] = metrics['test']['NDCG']
                    P[signature]    = metrics['test']['Precison']
                    R[signature]    = metrics['test']['Recall']
                    metrics_sum[signature]  = sum(NDCG[signature]) + sum(P[signature]) + sum(R[signature])
                    # write the results of the hyperparameter combinations searched so far
                    o_f.write(signature + '\n')
                    for k in range(len(K[signature])):
                        o_f.write('NDCG@%d = %.4f\n'%(K[signature][k], NDCG[signature][k]))
                    for k in range(len(K[signature])):
                        o_f.write('P@%d = %.4f\n'%(K[signature][k], P[signature][k]))
                    for k in range(len(K[signature])):
                        o_f.write('R@%d = %.4f\n'%(K[signature][k], R[signature][k]))
                    o_f.write('------------------------------------------\n')
                # find best hyperparameter set
                best_signature, best_sum = \
                    sorted(metrics_sum.items(), key=lambda x:x[1], reverse=True)[0]
                o_f.write('* best hyperparameter set\n')
                best_hp_values = best_signature.split(':')[1:]
                for i, value in enumerate(best_hp_values):
                    hp_name = hyperparameters[i]
                    hp_value = best_hp_values[i]
                    o_f.write('* {}: {}\n'.format(hp_name, hp_value))
                for k in range(len(K[best_signature])):
                    o_f.write('NDCG@%d = %.4f\n' %
                              (K[best_signature][k], NDCG[best_signature][k]))
                for k in range(len(K[best_signature])):
                    o_f.write('P@%d = %.4f\n' %
                              (K[best_signature][k], P[best_signature][k]))
                for k in range(len(K[best_signature])):
                    o_f.write('R@%d = %.4f\n' %
                              (K[best_signature][k], R[best_signature][k]))
                o_f.close()
            elif args.case_study:
                if args.model_dir is None:
                    initialize_model_directory(args)
                print ("model directory : %s" % args.model_dir)
                filename = "case_study.txt"
                if args.filename is not None:
                    filename = args.filename
                sys.stdout = open(os.path.join(args.model_dir, filename), "w")
                lf = construct_model(args)
                lf.cuda()
                case_study(lf)
            elif args.test_metrics:
                if args.model_dir is None:
                    initialize_model_directory(args)
                print ("model directory : %s" % args.model_dir)
                if args.filename is None:
                    if args.rollout_inference:
                        filename = "rollout"
                    else:
                        filename = "beam_search"
                    if args.reward_as_score:
                        filename += "_reward_as_score"
                    filename += ".txt"
                else:
                    filename = args.filename
                sys.stdout = open(os.path.join(args.model_dir, filename), "w")
                lf = construct_model(args)
                lf.cuda()
                test_metrics(lf)
            elif args.train:
                initialize_model_directory(args)
                lf = construct_model(args)
                lf.cuda()
                train(lf)

if __name__ == '__main__':
    run_experiment(args)
