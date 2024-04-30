# -*- coding: utf-8 -*-
"""Implementation of SOO (Munos, 2011)
"""

import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb
import Func
import torch
from BayeOpt import Baye


class SOO_node(P_node):
    def __init__(self, depth, index, parent, domain):
        super(SOO_node, self).__init__(depth, index, parent, domain)

        self.visited = False
        self.reward = -np.inf

    def update_reward(self, reward):
        self.reward = reward

    def get_reward(self):
        return self.reward

    def visit(self):
        self.visited = True

    def get_domain(self):
        return self.domain


class SOO(Algorithm):
    def __init__(self, n=100, h_max=100, domain=None, partition=None, K=3):
        super(SOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=SOO_node, K=K)

        self.iteration = 0
        self.n = n
        self.h_max = h_max

        self.v_max = -np.inf
        self.curr_depth = 0
        self.curr_node = None

        self.visit_num = 0

    def pull(self, time):
        self.iteration = time
        node_list = self.partition.get_node_list()
        max_value = -np.inf
        max_node = None
        depth = len(node_list) 
        while self.curr_depth <= min(self.partition.get_depth(), self.h_max):
            # print(self.curr_depth)
            # for node in node_list[self.curr_depth]:
            #     if node.get_reward() == -np.inf:
            #         node.visit()
            #         point = node.get_cpoint()
            #         domain = node.get_domain()
            #         point = torch.tensor(point).double().unsqueeze(0)
            #         self.visit_num += 1
            #         if method == "SOO":
            #             curr_reward = func(point)
            #         elif method == "mySOO":
            #             curr_reward = Baye(point, torch.tensor(domain).double(
            #             ), dim, func, kernel_type, init_num)

            #         node.update_reward(curr_reward)
            # print("curr_depth",self.curr_depth)
            for node in node_list[self.curr_depth]:  # for all node in the layer
                if (
                        node.get_children() is None
                ):  # if the node is not evaluated, evaluate it
                    if not node.visited:
                        node.visit()
                        self.curr_node = node
                        return node.get_cpoint(),node.get_domain()
                    if (
                            node.get_reward() >= max_value
                    ):  # find the leaf node with maximal reward
                        self.curr_node = node
                        max_value = node.get_reward()
                        max_node = node
            if max_value >= self.v_max:
                if max_node is not None:  # Found a leaf node
                    # print(self.curr_depth,self.partition.get_depth())
                    # print(max_node.get_depth())
                    self.partition.make_children(max_node, newlayer=(self.curr_depth >= self.partition.get_depth()))
                    # print(max_node.get_domain())
                    self.v_max = max_value
                    # return max_node.get_cpoint(), max_node.get_domain()
            self.curr_depth += 1
            # print("later_len",len(node_list))
            if self.curr_depth > min(
                    depth - 1, self.h_max
            ):  # if the search depth overflows, restart the loop
                # print("here")
                self.v_max = -np.inf
                self.curr_depth = 0
                max_node = None
                max_value = -np.inf
                depth = len(node_list)
                # return self.curr_node.get_cpoint(), self.curr_node.get_domain()

    # def pull(self, time, method, dim, func, kernel_type, init_num):

    #     self.iteration = time
    #     node_list = self.partition.get_node_list()

    #     while True:
    #         h = 0
    #         v_max = -np.inf
    #         while h <= min(self.partition.get_depth(), self.h_max):
    #             max_value = -np.inf
    #             max_node = None

    #             # for node in node_list[h]:
    #             #     if node.get_reward() == -np.inf:
    #             #         node.visit()
    #             #         point = node.get_cpoint()
    #             #         domain = node.get_domain()
    #             #         point = torch.tensor(point).double().unsqueeze(0)
    #             #         self.visit_num += 1
    #             #         if method == "SOO":
    #             #             curr_reward = func(point)
    #             #         elif method == "mySOO":
    #             #             curr_reward = Baye(point, torch.tensor(domain).double(
    #             #             ), dim, func, kernel_type, init_num)

    #             #         node.update_reward(curr_reward)

    #             for node in node_list[h]:  # for all node in the layer
    #                 if (
    #                     node.get_children() is None
    #                 ):  # if the node is not evaluated, evaluate it
    #                     if not node.visited:
    #                         node.visit()
    #                         self.curr_node = node
    #                         return node.get_cpoint(),node.get_domain()
    #                     if (
    #                         node.get_reward() >= max_value
    #                     ):  # find the leaf node with maximal reward
    #                         max_value = node.get_reward()
    #                         max_node = node
    #             if max_value >= v_max:
    #                 if max_node is not None:  # Found a leaf node
    #                     self.partition.make_children(
    #                         max_node, newlayer=(h >= self.partition.get_depth()))
    #                     v_max = max_value
    #                     # return max_node.get_cpoint(), max_node.get_domain()
    #             h += 1

    def receive_reward(self, time, reward):
        self.curr_node.update_reward(reward)

    def get_last_point(self):
        max_value = -np.inf
        max_node = None
        node_list = self.partition.get_node_list()
        # print(self.partition.get_depth())
        for h in range(len(node_list)):
            for node in node_list[h]:
                if node.get_reward() >= max_value:
                    max_value = node.get_reward()
                    max_node = node
        return max_node.get_cpoint(), max_value, self.partition.get_depth()

    def get_visit_num(self):
        return self.visit_num
    
    def get_tree(self):
        node_list = self.partition.get_node_list()
        return node_list
    
    def get_bo_point(self,curr_height):
        h = curr_height
        node_list = self.partition.get_node_list()
        if(h == -1):
           h = len(node_list) - 1

        max_node = None
        max_value = -np.inf
        while (1):
            if h <= (len(node_list) - 1) * 0.5 :
                h = len(node_list) - 1
            for node in node_list[h]:
                if node.get_children() is None: #未被扩展过
                    if node.get_reward() > max_value:
                       max_value = node.get_reward()
                       max_node = node
                       self.curr_node = max_node
            h -= 1
            if max_node is not None:    
                break
        
            
                
        max_node.update_children([1])
        return h, max_node.get_cpoint(), max_value, max_node.get_domain()


