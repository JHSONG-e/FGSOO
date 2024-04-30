from functions.functions import *
# from functions.mujoco_functions import *
from lamcts import MCTS
import argparse
import numpy as np
import torch
import time
import random
import os
import datetime



torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

seed = 0
budget = 200
num_exp = 10
init_nums = 40

func_name ='cassini'
target = Cassini2Gtopx
dim =22



func_val_all = torch.zeros(num_exp, budget)
func_val_all_full = torch.ones(num_exp, budget)
time_all = torch.zeros(num_exp)

folder = os.path.exists("./results")
if not folder:
    os.makedirs("./results")
path = "./results/" + func_name + "/LaMCTS_D" + str(dim)
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)


for seed in range(num_exp):
    random.seed(0)
    
    f = target(path, init_nums, dim)

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = MCTS(
        lb=f.lb,  # the lower bound of each problem dimensions
        ub=f.ub,  # the upper bound of each problem dimensions
        dims=f.dims,  # the problem dimensions
        ninits=f.ninits,  # the number of random samples used in initializations
        func=f,  # function object to be optimized
        Cp=f.Cp,  # Cp for MCTS
        leaf_size=f.leaf_size,  # tree leaf size
        kernel_type=f.kernel_type,  # SVM configruation
        gamma_type=f.gamma_type  # SVM configruation
    )

    start = time.time()
    agent.search(iterations=budget)
    end = time.time()
    time_all[seed] = end - start




    file_path = path + '/' + "seed" + str(seed) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__() + ".txt"

    file = open(str(file_path), 'w')
    file.write("=============================== \n")
    file.write("EX: LaMCTS \n")
    file.write("Datetime: " + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
    file.write("=============================== \n\n\n")
    file.write("=============================== \n")
    file.write("          BASIC INFOS           \n")
    file.write("=============================== \n")
    file.write("D: " + str(dim) + " \n")
    file.write("Init points: " + str(init_nums) + " \n")
    # file.write("x*:" + str(x_best) + "\n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    file.write("=============================== \n\n\n")


f = open(path + '/result' + str(budget))
for seed in range(num_exp):
    objectives = f.readline()
    objectives = np.array([float(i) for i in objectives[1: -2].split(', ')])
    func_val_all[seed] = -torch.from_numpy(np.minimum.accumulate(objectives))
    fX = -torch.from_numpy(objectives)
    func_val_all_full[seed] = fX


best_func_val = torch.zeros(budget)
for i in range(budget):
    best_func_val[i] = func_val_all[:, i].max()
mean = torch.mean(func_val_all, dim=0)
std = torch.sqrt(torch.var(func_val_all, dim=0))
median = torch.median(func_val_all, dim=0).values
# median = np.median(func_val_all.numpy(), axis=0)
file = open(str(path + '/experiment_result=' + str(round(-float(mean[-1]), 4)) + '.txt'), 'w')
file.write(f"The best function value across all the {num_exp} experiments: \n")
file.write(str(best_func_val))
file.write(f"\n\nThe mean of the function value across all the {num_exp} experiments: \n")
file.write(str(mean))
file.write(f"\n\nThe standard deviation of the function value across all the {num_exp} experiments: \n")
file.write(str(std))
file.write(f"\n\nThe median of the function value across all the {num_exp} experiments: \n")
file.write(str(median))
file.write(f"\n\nThe mean time each experiment consumes across all the {num_exp} experiments (s): \n")
file.write(str(time_all.mean()))
torch.save( - func_val_all_full, path + '/f.pt')


