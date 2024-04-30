import os
import numpy as np
import Func
import torch
import datetime
import time
import VOO



    
budget = 200
num_exp = 10
method = 'VOO'


function_name ='cassini'
func = Func.cassini2_gtopx
dim = 22
domain = [[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
        1.05, 1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
        [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
        6.5, 291.0, torch.pi, torch.pi, torch.pi, torch.pi]]
domain = [list(row) for row in zip(*domain)]              #Gtopx

domain = np.asarray(domain)

result = np.zeros((num_exp, budget))


folder = os.path.exists("./results")
if not folder:
    os.makedirs("./results")
path = "./results/" + function_name + "/" + str(method) + "_D" + str(dim)
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

time_all = torch.zeros(num_exp)

for seed in range(num_exp):
    np.random.seed(seed)
    max_try = 5000
    init_num = 5

    print("exp_num=",seed)
    file_path = path + '/' + "log_seed" + str(seed) + ".txt"
    file = open(str(file_path), 'w')
    
    x_data,y_data = np.zeros((0,dim)),np.zeros((0,1))
    
    start = time.time() 
    x_evals = VOO.x_sampler(n_sample=init_num,x_minmax=domain)
    y_evals = [func(torch.from_numpy(x_eval)) for x_eval in x_evals] 
    y_evals = [y.numpy() for y in y_evals]
    if y_evals[0].shape ==(1,1):
        y_evals = [ y[0] for y in y_evals]
    
    x_evals = np.asarray(x_evals)[:,0,:] # reshape

    x_data,y_data = np.concatenate((x_data,x_evals),axis=0),np.concatenate((y_data,y_evals),axis=0) # concatenate
    x_sol,y_sol = VOO.get_best_xy(x_data,y_data)

    for i in range(init_num):
        result[seed, i] = y_sol

    for it in range(budget - init_num):
        x_evals = VOO.sample_from_best_voronoi_cell(
        x_data, y_data, domain, max_try_sbv=max_try)
        y_evals = [func(torch.from_numpy(x_eval)) for x_eval in x_evals] 
        y_evals = [y.numpy() for y in y_evals]
        if y_evals[0].shape ==(1,1):
            y_evals = [ y[0] for y in y_evals]
        x_evals = np.asarray(x_evals)[:,0,:] # reshape
        x_data,y_data = np.concatenate((x_data,x_evals),axis=0),np.concatenate((y_data,y_evals),axis=0) # concatenate
        x_sol,y_sol = VOO.get_best_xy(x_data,y_data)
        # print(x_sol,y_sol)
        result[seed, it + init_num] = y_sol
        
        file.write("iter:"+str(it+init_num) + "\n")
        file.write("sel_point:" + str(x_evals) +"\n")
        file.write("sel_y:" + str(y_evals) +"\n\n")
        file.write("best_x:" + str(x_sol)+"\n")
        file.write("best_y:"+str(y_sol)+"\n")

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
    
    
    file.write("Budget: " + str(budget) + " \n")

    file.write("Objective Function: " + function_name + " \n")
    file.write("Random seed: " + str(seed) + " \n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    file.write("x*:" + str(x_sol) + "\n")
    file.write("optimal value:" + str( -y_sol ) + "\n")
    file.write("=============================== \n\n\n")

result = torch.from_numpy(result)
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
