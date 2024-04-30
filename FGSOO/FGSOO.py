import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb
import Func 
import torch
from BayeOpt import Baye


class FGSOO_node(P_node):
    def __init__(self, depth, index, parent, domain, data = None):
        super(FGSOO_node, self).__init__(depth, index, parent, domain)

        self.visited = False
        self.reward = -np.inf
        self.ucb_value = -np.inf
        self.data = data
        self.best_x = None

    def update_reward(self, dataset,reward,ucb_value):
        self.reward = reward
        self.data = dataset
        self.ucb_value = ucb_value

    def get_reward(self):
        return self.reward

    def visit(self):
        self.visited = True

    def get_domain(self):
        return self.domain
    
    def get_UCB_value(self):
        return self.ucb_value
    
    def update_data(self, data):
        self.data = data
    
    def update_bestx(self, x):
        self.best_x = x


class FGSOO(Algorithm):
    def __init__(self, n=100, h_max=100, domain=None, partition=None, K=3):
        super(FGSOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=FGSOO_node, K=K)

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

        max_node = None
        max_value = -np.inf
        flag = False
        h = 0
        while h <= self.partition.get_depth():
            for node in node_list[h]:
                if node.get_children() is None:
                    if not node.visited:
                        node.visit()
                        self.curr_node = node
                        flag = True
                        return flag,node,node.get_cpoint(),node.get_domain()
            
            h += 1
        return flag,node,node.get_cpoint(),node.get_domain()
    
    def likeDOO_partition(self, time, dim, func, kernel_type, init_num, itr, re_budget):
        self.iteration = time
        node_list = self.partition.get_node_list()
        h = 0
        cost_budget = 0
        diff_h_num = 0
        diff_h_node = []
        for h in range(len(node_list)):  #先判断只是同层评估
            max_node = None
            max_value = -np.inf
            for node in node_list[h]:
                if node.get_children() is None:
                    if node.get_reward() >= max_value:
                        max_value = node.get_reward()
                        max_node = node
            if max_node is not None:
                diff_h_num +=1
                diff_h_node.append(max_node)
        
        if(diff_h_num == 1):
            self.partition.make_children(max_node, newlayer=(max_node.get_depth() >= self.partition.get_depth()))
            return cost_budget,max_node,max_node.get_cpoint(),max_node.get_domain()
        
        else:    #如果不是同层评估
            maxUCB_node = None
            maxUCB_value = -np.inf
            for node in diff_h_node:
                # re_budget -= cost_budget
                if node.get_UCB_value() == -np.inf:
                    point = node.get_cpoint()
                    point = torch.tensor(point,dtype=torch.float64).unsqueeze(0)
                    curr_domain = node.get_domain()
                    if re_budget < itr :
                        itr = re_budget
                    dataset,reward,ucb_value, best_x=Baye(point,torch.tensor(curr_domain,dtype=torch.float64), dim, func, kernel_type,  node.data, itr,init_num )
                    node.update_reward(dataset,reward,ucb_value)
                    node.update_bestx(best_x)
                    cost_budget +=itr
                    re_budget = re_budget - itr
                if node.get_UCB_value() >= maxUCB_value:
                    maxUCB_node = node
                    maxUCB_value = node.get_UCB_value()
            self.partition.make_children(maxUCB_node, newlayer=(maxUCB_node.get_depth() >= self.partition.get_depth()))
            return cost_budget,maxUCB_node,maxUCB_node.get_cpoint(),maxUCB_node.get_domain()

        

    def receive_reward(self, dataset, reward, ucb_value):
        self.curr_node.update_reward(dataset,reward,ucb_value)

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

