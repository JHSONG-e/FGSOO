import numpy as np
import torch
import time
import random
import os
import datetime
f = open('results/cassini/LaMCTS_D22_de20_0521-21-40-13/result200')
num_exp = 20
dim_high = 22
d_e = 20
seed = 0
budget = 200
init_nums = 5
func_val_all = torch.zeros(num_exp, budget)
func_val_all_full = torch.ones(num_exp, budget)

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
# file = open(str('./results1/experiment_result=' + str(round(-float(mean[-1]), 4)) + '.txt'), 'w')
# file.write(f"The best function value across all the {num_exp} experiments: \n")
# file.write(str(best_func_val))
# file.write(f"\n\nThe mean of the function value across all the {num_exp} experiments: \n")
# file.write(str(mean))
# file.write(f"\n\nThe standard deviation of the function value across all the {num_exp} experiments: \n")
# file.write(str(std))
# file.write(f"\n\nThe median of the function value across all the {num_exp} experiments: \n")
# file.write(str(median))
# file.write(f"\n\nThe mean time each experiment consumes across all the {num_exp} experiments (s): \n")
# file.write(str(time_all.mean()))
torch.save(func_val_all_full, './results1/f2.pt')
# print(func_val_all_full)