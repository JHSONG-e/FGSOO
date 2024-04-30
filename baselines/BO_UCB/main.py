import os
import numpy as np

import Func
import torch
import BayeOpt
import datetime
import time


budget = 200
num_exp = 10
method = 'BO_UCB'

function_name ='cassini'
func = Func.cassini2_gtopx
dim = 22
domain = [[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
        1.05, 1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
        [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
        6.5, 291.0, torch.pi, torch.pi, torch.pi, torch.pi]]
domain = [list(row) for row in zip(*domain)]              #Gtopx




kernel_type = 'matern'
init_num = 5
pop_size = 5


result = torch.zeros(num_exp, budget)


folder = os.path.exists("./results")
if not folder:
    os.makedirs("./results")
path = "./results/" + function_name + "/" + str(method) + "_D" + str(dim)
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

time_all = torch.zeros(num_exp)



for seed in range(num_exp):
    torch.manual_seed(seed)
    print("exp_num=",seed)
    file_path = path + '/' + "log_seed" + str(seed) + ".txt"
    file = open(str(file_path), 'w')
    bounds = torch.tensor(domain,dtype=torch.float64)

    start = time.time()

    bounds_nor = torch.tensor([[0., 1.]] * dim,dtype = torch.float64)

    dataset = BayeOpt.init_points_dataset_bo(
    init_num,bounds_nor, bounds, func)


    for i in range(budget):
        if i < init_num + 1:
            result[seed,i] = dataset['f'].max()
            file.write("iter:" +str(i) +"\n")
            file.write("reward:"+ str(dataset['f'].max())+"\n\n")
        else:
            _, next_x = BayeOpt.next_point_bo(dataset, 0.2 * dim * torch.log(torch.tensor(2 * dataset['f'].shape[0])) , bounds_nor, kernel_type)
            next_y = next_x * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
            next_f = func(next_y).reshape(1, 1)
            print("f:",next_f)
            dataset = BayeOpt.update_dataset_ucb(next_y, next_x, next_f, dataset)
            result[seed,i] = dataset['f'].max()
            file.write("iter:" +str(i) +"\n")
            file.write("reward:"+ str(next_f)+"\n\n")
    
    max_value = dataset['f'].max()
    end = time.time()
    time_all[seed] = end - start


    file_path = path + '/' + "seed" + str(seed) + ".txt"
    file = open(str(file_path), 'w')

    file.write("=============================== \n")
    file.write("Optimization method: " + str(method) + " \n")
    file.write("Datetime: " +
            str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
    file.write("=============================== \n\n\n")
    file.write("=============================== \n")
    file.write("          BASIC INFOS           \n")
    file.write("=============================== \n")
    file.write("dimension: " + str(dim) + " \n")
    
    file.write("Kernel method: " + str(kernel_type) + " \n")
    
    file.write("Budget: " + str(budget) + " \n")
    file.write("Objective Function: " + function_name + " \n")
    file.write("Init points: " + str(init_num) + " \n")
    file.write("Random seed: " + str(seed) + " \n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    # file.write("x*:" + str(last_point) + "\n")
    file.write("optimal value:" + str( -max_value ) + "\n")
    file.write("=============================== \n\n\n")

best_func_val = torch.zeros(budget)
for i in range(budget):
    best_func_val[i] = result[:, i].max()
mean = torch.mean(result, dim=0)
std = torch.sqrt(torch.var(result, dim=0))
median = torch.median(result, dim=0).values
# median = np.median(func_val_all.numpy(), axis=0)
file = open(str(path + '/experiment_result=' +
            str(round(-float(mean[-1]), 4)) + '.txt'), 'w')
file.write(f"The best function value across all the {num_exp} experiments: \n")
file.write(str(best_func_val))
file.write(
    f"\n\nThe mean of the function value across all the {num_exp} experiments: \n")
file.write(str(mean))
file.write(
    f"\n\nThe standard deviation of the function value across all the {num_exp} experiments: \n")
file.write(str(std))
file.write(
    f"\n\nThe median of the function value across all the {num_exp} experiments: \n")
file.write(str(median))
file.write(
    f"\n\nThe mean time each experiment consumes across all the {num_exp} experiments (s): \n")
file.write(str(time_all.mean()))
torch.save(-result, path + '/f.pt')
torch.save(-mean, path + '/f_mean.pt')
print(-mean)