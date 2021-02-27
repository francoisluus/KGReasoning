#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
from scipy import stats
import os

eps = 1e-6

def Identity(x):
    return x

class BoxOffsetIntersection(nn.Module):
    
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0) 
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding

class BetaIntersection(nn.Module):

    def __init__(self, dim, use_att):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.use_att = use_att
        
        if use_att:
            self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
            self.layer2 = nn.Linear(2 * self.dim, self.dim)

            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        if self.use_att:
            all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
            layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
            attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)

            alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
            beta_embedding = torch.sum(attention * beta_embeddings, dim=0)
        else:
            num_conj = alpha_embeddings.shape[0]
            alpha_embedding = torch.sum(alpha_embeddings, dim=0) / num_conj
            beta_embedding = torch.sum(beta_embeddings, dim=0) / num_conj

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

def order_bounds(embedding): # ensure lower < upper truth bound for logic embedding
    embedding = torch.clamp(embedding, 0, 1)
    lower, upper = torch.chunk(embedding, 2, dim=-1)
    contra = lower > upper
    if contra.any(): # contradiction
        mean = (lower + upper) / 2
        lower = torch.where(lower > upper, mean, lower)
        upper = torch.where(lower > upper, mean, upper)
    ordered_embedding = torch.cat([lower, upper], dim=-1)
    return ordered_embedding

def valclamp(x, a=1, b=6, lo=0, hi=1): # relu1 with gradient-transparent clamp on negative
    elu_neg = a * (torch.exp(b * x) - 1)
    return ((x < lo).float() * (lo + elu_neg - elu_neg.detach()) +
            (lo <= x).float() * (x <= hi).float() * x + 
            (hi < x).float())
    
class LogicIntersection(nn.Module):

    def __init__(self, dim, tnorm, bounded, use_att, use_gtrans):
        super(LogicIntersection, self).__init__()
        self.dim = dim
        self.tnorm = tnorm
        self.bounded = bounded
        self.use_att = use_att
        self.use_gtrans = use_gtrans # gradient transparency
        
        if use_att: # use attention with weighted t-norm
            self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)

            if bounded:
                self.layer2 = nn.Linear(2 * self.dim, self.dim) # same weight for bound pair
            else:
                self.layer2 = nn.Linear(2 * self.dim, 2 * self.dim)

            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        if self.use_att: # use attention with weighted t-norm
            layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, batch_size, 2 * dim)
            attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)
            attention = attention / torch.max(attention, dim=0, keepdim=True).values

            if self.bounded: # same weight for bound pair
                attention = torch.cat([attention, attention], dim=-1)

            if self.tnorm == 'mins': # minimum / Godel t-norm
                smooth_param = -10  # smooth minimum
                min_weights = attention * torch.exp(smooth_param * embeddings)
                embedding = torch.sum(min_weights * embeddings, dim=0) / torch.sum(min_weights, dim=0)
                if self.bounded:
                    embedding = order_bounds(embedding)
                
            elif self.tnorm == 'luk': # Lukasiewicz t-norm
                embedding = 1 - torch.sum(attention * (1 - embeddings), dim=0)
                if self.use_gtrans:
                    embedding = valclamp(embedding, b=6./embedding.shape[0])
                else:
                    embedding = torch.clamp(embedding, 0, 1)
                    
            elif self.tnorm == 'prod': # product t-norm
                embedding = torch.prod(torch.pow(torch.clamp(embeddings, 0, 1) + eps, attention), dim=0)
                
        else: # no attention
            if self.tnorm == 'mins': # minimum / Godel t-norm
                smooth_param = -10  # smooth minimum
                min_weights = torch.exp(smooth_param * embeddings)
                embedding = torch.sum(min_weights * embeddings, dim=0) / torch.sum(min_weights, dim=0)
                if self.bounded:
                    embedding = order_bounds(embedding)
                
            elif self.tnorm == 'luk': # Lukasiewicz t-norm
                embedding = 1 - torch.sum(1 - embeddings, dim=0)
                if self.use_gtrans:
                    embedding = valclamp(embedding, b=6./embedding.shape[0])
                else:
                    embedding = torch.clamp(embedding, 0, 1)
                    
            elif self.tnorm == 'prod': # product t-norm
                embedding = torch.prod(embeddings, dim=0)
        
        return embedding

class LogicProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, num_layers, bounded):
        super(LogicProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bounded = bounded
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = torch.sigmoid(x)
        
        if self.bounded:
            lower, upper = torch.chunk(x, 2, dim=-1)
            upper = lower + upper * (1 - lower)
            x = torch.cat([lower, upper], dim=-1)
            
        return x

class SizePredict(nn.Module):
    def __init__(self, entity_dim):
        super(SizePredict, self).__init__()
        
        self.layer2 = nn.Linear(entity_dim, entity_dim // 4)
        self.layer1 = nn.Linear(entity_dim // 4, entity_dim // 16)
        self.layer0 = nn.Linear(entity_dim // 16, 1)

        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer0.weight)

    def forward(self, entropy_embedding):
        x = self.layer2(entropy_embedding)
        x = F.relu(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer0(x)
        x = torch.sigmoid(x)

        return x.squeeze()

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 geo, test_batch_size=1,
                 box_mode=None, logic_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        
        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # centor for entities
            activation, cen = box_mode
            self.cen = cen # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # center for entities
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2)) # alpha and beta
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
        elif self.geo == 'logic':
            self.tnorm, self.bounded, use_att, use_gtrans, hidden_dim, num_layers = logic_mode
            if self.bounded:
                lower = torch.rand((nentity, self.entity_dim))
                upper = lower + torch.rand((nentity, self.entity_dim)) * (1 - lower)
                self.entity_embedding = nn.Parameter(torch.cat([lower, upper], dim=-1))
            else:
                self.entity_embedding = nn.Parameter(torch.rand((nentity, self.entity_dim * 2)))
        
        if self.geo in ['box', 'vec', 'beta']:
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding, 
                a=0., 
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            self.use_att, hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim, self.use_att)
            self.projection_net = BetaProjection(self.entity_dim * 2, 
                                             self.relation_dim, 
                                             hidden_dim, 
                                             self.projection_regularizer, 
                                             num_layers)
        elif self.geo == 'logic':
            tnorm, bounded, use_att, use_gtrans, hidden_dim, num_layers = logic_mode
            self.center_net = LogicIntersection(self.entity_dim, tnorm, bounded, use_att, use_gtrans)
            self.projection_net = LogicProjection(self.entity_dim * 2, 
                                             self.relation_dim, 
                                             hidden_dim, 
                                             num_layers,
                                             bounded)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'logic':
            return self.forward_logic(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def embed_query_beta(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_entropy = None
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                     query_structure), 
                                          self.transform_union_structure(query_structure), 
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure], 
                                                                           query_structure, 
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)

            if positive_sample is None: # test step - measure entropy
                all_entropy = all_dists.entropy()

        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, all_entropy

    def embed_query_logic(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using logic embeddings
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_logic(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    if self.bounded:
                        lower_embedding, upper_embedding = torch.chunk(embedding, 2, dim=-1)
                        embedding = torch.cat([1 - upper_embedding, 1 - lower_embedding], dim=-1)
                    else:
                        embedding = 1 - embedding
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_logic(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def cal_logit_logic(self, entity_embedding, query_embedding):
        if self.bounded:
            lower_embedding, upper_embedding = torch.chunk(entity_embedding, 2, dim=-1)
            query_lower_embedding, query_upper_embedding = torch.chunk(query_embedding, 2, dim=-1)
            
            lower_dist = torch.norm(lower_embedding - query_lower_embedding, p=1, dim=-1)
            upper_dist = torch.norm(query_upper_embedding - upper_embedding, p=1, dim=-1)
            
            logit = self.gamma - (lower_dist + upper_dist) / 2 / lower_embedding.shape[-1]
        else:
            logit = self.gamma - torch.norm(entity_embedding - query_embedding, p=1, dim=-1) / query_embedding.shape[-1]
            
        logit *= 100
        
        return logit

    def forward_logic(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_entropy = None
        all_idxs, all_embeddings = [], []
        all_union_idxs, all_union_embeddings = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                embedding, _ = \
                    self.embed_query_logic(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                     query_structure), 
                                          self.transform_union_structure(query_structure), 
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_embeddings.append(embedding)
            else:
                embedding, _ = self.embed_query_logic(batch_queries_dict[query_structure], 
                                                                           query_structure, 
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_embeddings.append(embedding)

        if len(all_embeddings) > 0:
            all_embeddings = torch.cat(all_embeddings, dim=0).unsqueeze(1)

            if positive_sample is None and self.bounded: # test step - measure entropy
                lower, upper = torch.chunk(all_embeddings, 2, dim=-1)
                truth_interval = upper - lower
                distribution = torch.distributions.uniform.Uniform(lower, upper + eps)
                all_entropy = (distribution.entropy(), truth_interval)

        if len(all_union_embeddings) > 0:
            all_union_embeddings = torch.cat(all_union_embeddings, dim=0).unsqueeze(1)
            all_union_embeddings = all_union_embeddings.view(all_union_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_logic(positive_embedding, all_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs] # positive samples for union queries in this batch
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_logic(positive_embedding, all_union_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_logic(negative_embedding, all_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_logic(negative_embedding, all_union_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, all_entropy

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                    query_structure), 
                                         self.transform_union_structure(query_structure), 
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure], 
                                                                             query_structure, 
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        else:
            all_offset_embeddings = None
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, all_offset_embeddings
    
    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                    query_structure), 
                                                                self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, None

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        
        if model.geo == 'logic':
            entity_embedding = model._parameters['entity_embedding'].data
            if model.bounded:
                model._parameters['entity_embedding'].data = order_bounds(entity_embedding)
            else:
                model._parameters['entity_embedding'].data = torch.clamp(entity_embedding, 0, 1)
        
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'positive_score_mean': positive_score.mean(),
            'negative_score_mean': negative_score.mean(),
            'positive_logit_mean': positive_logit.mean(),
            'negative_logit_mean': negative_logit.mean(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs, query_entropy = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    entropy = None
                    truth_interval = None
                    entropy_emb = None
                    truth_interval_emb = None
                    
                    if query_entropy is not None:
                        if model.geo == 'box':
                            if idx < query_entropy.shape[0]:
                                entropy_emb = query_entropy[idx].squeeze()
                                entropy = torch.norm(query_entropy[idx], p=1).item()
                        elif model.geo == 'beta':
                            if idx < query_entropy.shape[0]:
                                entropy_emb = query_entropy[idx].squeeze()
                                entropy = query_entropy[idx].sum().item()
                        elif model.geo == 'logic' and model.bounded:
                            entropy, truth_interval = query_entropy
                            if idx < entropy.shape[0]:
                                entropy_emb = entropy[idx].squeeze()
                                entropy = entropy[idx].sum().item()
                                truth_interval_emb = truth_interval[idx].squeeze()
                                truth_interval = torch.norm(truth_interval[idx], p=1).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'entropy': entropy,
                        'truth_interval': truth_interval,
                        'entropy-MAE': entropy_emb,
                        'truth_interval-MAE': truth_interval_emb,
                        'num_hard_answer': num_hard,
                        'num_answer': num_easy + num_hard
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['entropy-MAE', 'truth_interval-MAE']:
                    continue
                                
                if metric in ['entropy', 'truth_interval']:
                    xn, yn = [], []
                    for log in logs[query_structure]:
                        if log[metric] is not None:
                            xn.append(log[metric])
                            yn.append(log['num_answer'])
                    if len(xn):
                        metrics[query_structure][metric + '-Spearman'] = stats.spearmanr(xn, yn)[0]
                        metrics[query_structure][metric + '-Pearson'] = stats.pearsonr(xn, yn)[0]
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        for metric in ['entropy-MAE', 'truth_interval-MAE']:
            epochs = 250
            size_scale = 1000.
            train_in = []
            train_out = []
            test_in = {}
            test_out = {}
            for query_structure in logs:
                if metric in logs[query_structure][0].keys():
                    samples = []
                    for log in logs[query_structure]:
                        if log[metric] is not None:
                            sample_out = torch.clamp(torch.tensor(log['num_answer'] / size_scale), 0, 1)
                            if args.cuda:
                                sample_out = sample_out.cuda()
                            samples.append((log[metric], sample_out))
                    if len(samples) > 0:
                        random.shuffle(samples)
                        train_in.extend([_[0] for _ in samples[::2]])
                        train_out.extend([_[1] for _ in samples[::2]])
                        test_in[query_structure] = torch.stack([_[0] for _ in samples[1::2]])
                        test_out[query_structure] = torch.stack([_[1] for _ in samples[1::2]])

            if len(train_in) > 0:
                train_in = torch.stack(train_in)
                train_out = torch.stack(train_out)
                in_mean = train_in.mean()
                train_in /= in_mean
                
                net = SizePredict(model.entity_dim)
                if args.cuda:
                    net = net.cuda()
                optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

                for i in range(epochs):
                    optimizer.zero_grad()
                    prediction = net(train_in)
                    error = (prediction - train_out) / train_out
                    loss = error.abs().mean()
                    loss.backward()
                    optimizer.step()

                for query_structure in logs:
                    if query_structure in test_in:
                        prediction = net(test_in[query_structure]/in_mean)
                        error = (prediction - test_out[query_structure]) / test_out[query_structure]
                        metrics[query_structure][metric] = error.abs().mean()
                                
        return metrics