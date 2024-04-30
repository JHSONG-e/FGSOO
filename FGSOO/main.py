import os
import numpy as np
from partition import KaryPartition
from FGSOO import FGSOO
import Func
import torch
from BayeOpt import Baye
import datetime
import time


k = 2
dim = 15
budget = 200
num_exp = 10
method = 'FGSOO'
BO_iteration = 20

function_name = 'cassini'
func = Func.cassini2_gtopx
dim = 22
domain = [[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.05, 1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
            [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
            6.5, 291.0, torch.pi, torch.pi, torch.pi, torch.pi]]
domain = [list(row) for row in zip(*domain)]              #Gtopx


partition = KaryPartition
kernel_type = 'matern'
init_num = 4

result = torch.zeros(num_exp, budget)


folder = os.path.exists("./FGSOO/results_d15_k2_20")
if not folder:
    os.makedirs("./FGSOO/results_d15_k2_20")
path = "./FGSOO/results_d15_k2_20/" + function_name + "/" + str(method) + "_D" + str(dim)
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

time_all = torch.zeros(num_exp)



for seed in range(num_exp):
    algo = FGSOO(n=budget, h_max=200, domain=domain, partition=partition, K=k)
    
    torch.manual_seed(seed)
    print(seed)
    
    file_path = path + '/' + "log_seed" + str(seed) + ".txt"
    file = open(str(file_path), 'w')
    end_i = 0
    begin_i = 0
    start = time.time()
    cnt = 0 
    while (begin_i < budget):
        # print(begin_i)
        # print(rate/dim)
        # min_dim = np.argmin(curr_diff_value)
        flag, node, point, curr_domain = algo.pull(begin_i)
        tensor_point = torch.tensor(point,dtype=torch.float64).unsqueeze(0)
        tensor_domain = torch.tensor(curr_domain,dtype=torch.float64)
        # print("begin_i:",begin_i)
        if flag:  #如果未被评估过，则评估一次
            nor_x = (tensor_point - tensor_domain.t()[0]) / (tensor_domain.t()[1] - tensor_domain.t()[0]) 
            nor_x = torch.where(torch.isnan(nor_x), torch.tensor(0.5), nor_x)
            # if function_name =='chembl':
            #     nor_x[:,0]=0.5
            # init_x = (point - bounds.t()[0]) * 2 / (bounds.t()[1] - bounds.t()[0]) - 1
            reward = func(tensor_point).reshape(1, 1)
            new_dataset = node.data
            if new_dataset is None or new_dataset['x'].shape[0] == 0:
                new_dataset = {
            'x':nor_x,
            'y':tensor_point,
            'f':reward
        }   
            else:
                new_dataset['x'] = torch.cat((new_dataset['x'],nor_x))
                new_dataset['y'] = torch.cat((new_dataset['y'],tensor_point))
                new_dataset['f'] = torch.cat((new_dataset['f'],reward))
            
            algo.receive_reward(new_dataset,reward,-np.inf)
            end_i +=1
            file.write("point:" + str(point) +"\n")
            file.write("curr_domain:" + str(curr_domain) +"\n")
            file.write("reward:" + str(reward) +"\n\n")
        
        else:     #全部评估过，则选择一个node分裂
            cost, node, point, curr_domain = algo.likeDOO_partition(begin_i,dim,func,kernel_type,init_num,BO_iteration,budget - begin_i)
            reward = node.get_reward()

            file.write("Selected point: \n")
            file.write("point:" + str(point) +"\n")
            file.write("height of point:" + str(node.get_depth())+"\n")
            file.write("curr_domain:" + str(curr_domain) +"\n")
            file.write("reward:" + str(reward) +"\n")

            if cost != 0:  #如果不是同层进行分裂
                ucb_value = node.get_UCB_value()
                file.write("UCB_value:" + str(ucb_value)+ "\n")
                end_i += cost
            
            file.write("\n")


        last_point, max_value,max_depth= algo.get_last_point()
        
        max_value = max_value.unsqueeze(0)

        for i in range(begin_i,end_i):
            result[seed, i] = max_value
        begin_i = end_i
        
    end = time.time()
    time_all[seed] = end - start

    file_path = path + '/' + "tree" + str(seed) + ".txt"
    file = open(str(file_path), 'w')
    node_list = algo.get_tree()
    for h in range(len(node_list)):
        file.write("height:" + str(h)+ "\n")
        for node in node_list[h]:
            file.write(str(node.get_cpoint())+":")
            file.write(str(node.get_reward())+"\n")
        file.write("\n\n")
    

    # print(algo.get_visit_num())
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
    file.write("BO_iter:" + str(BO_iteration)+" \n")
    file.write("Kernel method: " + str(kernel_type) + " \n")
    
    file.write("Budget: " + str(budget) + " \n")
    file.write("K-ary: " + str(k) + " \n")
    file.write("Objective Function: " + function_name + " \n")
    file.write("Init points: " + str(init_num) + " \n")
    file.write("Random seed: " + str(seed) + " \n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    file.write("x*:" + str(last_point) + "\n")
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



