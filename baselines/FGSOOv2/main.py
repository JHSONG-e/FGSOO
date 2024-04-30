import os
import numpy as np
from partition import KaryPartition
from SOO import SOO
import Func
import torch
from BayeOpt import Baye
import datetime
import time




k = 2
dim = 15
budget = 200
num_exp = 10
method = 'FGSOOv2'
BO_iteration = 20
end_SOO = 100

function_name ='cassini'
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


folder = os.path.exists("./results")
if not folder:
    os.makedirs("./results")
path = "./results/" + function_name + "/" + str(method) + "_D" + str(dim)
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

time_all = torch.zeros(num_exp)



for seed in range(num_exp):
    algo = SOO(n=budget, h_max=200, domain=domain, partition=partition, K=k)
    torch.manual_seed(seed)
    print("exp_num=",seed)
    file_path = path + '/' + "log_seed" + str(seed) + ".txt"
    file = open(str(file_path), 'w')
    end_i = 0
    begin_i = 0
    start = time.time()
    cnt = 0 
    h = -1
    while (begin_i < budget):
        if method =='FGSOOv2':
            if( begin_i < end_SOO ):
                point, curr_domain = algo.pull(begin_i)
                point = torch.tensor(point,dtype=torch.float64).unsqueeze(0)
                reward = func(point)
                end_i += 1
            else:
                h, point, value,curr_domain = algo.get_bo_point(h)
                point = torch.tensor(point,dtype=torch.float64).unsqueeze(0)
                reward = Baye(point,value,torch.tensor(curr_domain,dtype=torch.float64), dim, func, kernel_type,BO_iteration, init_num )
                    
                end_i += BO_iteration

        
        algo.receive_reward(end_i,reward)

        file.write(str(point) +"\n")
        file.write(str(curr_domain) +"\n")
        file.write(str(reward) +"\n\n")
        last_point, max_value,max_depth= algo.get_last_point()
        # print(last_point)
        max_value = max_value.unsqueeze(0)
        # print(res.shape)
        for i in range(begin_i,end_i):
            result[seed, i] = max_value
        begin_i = end_i
    end = time.time()
    time_all[seed] = end - start
    # print(max_depth)



    file_path = path + '/' + "tree" + str(seed) + ".txt"
    file = open(str(file_path), 'w')
    node_list = algo.get_tree()
    for h in range(len(node_list)):
        file.write("height:" + str(h)+ "\n")
        for node in node_list[h]:
            file.write(str(node.get_reward())+" ")
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
    
    file.write("Kernel method: " + str(kernel_type) + " \n")
    
    file.write("Budget: " + str(budget) + " \n")
    file.write("End budget of SOO: " + str(end_SOO) + " \n")
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


