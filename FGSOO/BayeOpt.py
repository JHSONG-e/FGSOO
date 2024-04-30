import torch
import Func
import botorch
import math
import numpy as np
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils import standardize
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.kernels.piecewise_polynomial_kernel import PiecewisePolynomialKernel
from gpytorch.kernels.rq_kernel import RQKernel
from gpytorch.kernels.polynomial_kernel import PolynomialKernel
from pyDOE import lhs

def upper_confidence_bound(bounds, model, beta):
    """
    Basic UCB acquisition function
    :param bounds: the objective function containing the bounds and dimension information
    :param model: the Gaussian Process model, find next point according to this model and UCB
    :param beta: UCB parameter
    :return: next point found by UCB and cost time
    """
    next_point, acq_value_list = optimize_acqf(
        acq_function=UpperConfidenceBound(model=model, beta=beta),
        bounds=bounds.t(),
        q=1,
        num_restarts=10,
        raw_samples=512
    )
    return next_point


def init_points_dataset_bo(n, bounds_nor,bounds, objective_function):
    """
    random init n points for basic BO
    :param n: init points num
    :param objective_function: the objective function containing the bounds and dimension information
    :return: the original dataset (a dict have 3 elements 'x', 'y' and 'f')
    """
    dim = bounds.shape[0]

    # init_point = torch.from_numpy(lhs(dim, n)).type(torch.double)
    # 原始
    # dataset = {'x': init_point}
    # dataset = {'x': torch.rand(n, dim) * (bounds_nor.t()
    #                                   [1] - bounds_nor.t()[0]) + bounds_nor.t()[0]}
    dataset = {'x': torch.rand(n, dim, dtype=torch.float64) }
    # # 规范化
   
    dataset['y'] = dataset['x'] * (bounds.t()[1] - bounds.t()[0])  + bounds.t()[0]
    # dataset['y'] = (dataset['x'] + 1) * (bounds.t()[1] - bounds.t()[0]) / 2   + bounds.t()[0]
    # dataset = {'y': torch.rand(n, dim) * (bounds.t()
    #                                       [1] - bounds.t()[0]) + bounds.t()[0]}

    dataset['f'] = objective_function(dataset['y']).reshape(n, 1)
    # print(dataset)
    return dataset


def fit_model_gp(dataset, kernel_type):
    """
    Use training dataset to fit the Gaussian Process Model
    :param dataset: a dict have 2 elements 'x' and 'f', each of them is a tensor shaped (n, dim) and (n, 1)
    :return: the GP model, the marginal log likelihood and cost time
    """
    dataset_x = dataset['x'].clone()
    dataset_f = dataset['f'].clone()
    
    # print(dataset_x,dataset_f)
    mean, std = dataset_f.mean(), dataset_f.std()
    # std = 1e-5 if std < 1e-5 else std
    std = 1.0 if std < 1e-6 else std
    dataset_f = (dataset_f - mean) / std
    # dataset_f = standardize(dataset_f)
    if kernel_type == 'rbf':
        model = SingleTaskGP(dataset_x, dataset_f, covar_module=ScaleKernel(
            RBFKernel(ard_num_dims=dataset_x.shape[-1])))
    elif kernel_type == 'matern':
        model = SingleTaskGP(dataset_x, dataset_f)
    elif kernel_type == 'linear':
        model = SingleTaskGP(dataset_x, dataset_f,
                             covar_module=ScaleKernel(LinearKernel()))
    elif kernel_type == 'piece_poly':
        model = SingleTaskGP(dataset_x, dataset_f,
                             covar_module=ScaleKernel(PiecewisePolynomialKernel(ard_num_dims=dataset_x.shape[-1], q=3)))
    elif kernel_type == 'rq':
        model = SingleTaskGP(dataset_x, dataset_f, covar_module=ScaleKernel(
            RQKernel(ard_num_dims=dataset_x.shape[-1])))
    elif kernel_type == 'poly':
        model = SingleTaskGP(dataset_x, dataset_f,
                             covar_module=ScaleKernel(PolynomialKernel(power=2, ard_num_dims=dataset_x.shape[-1])))
    else:
        print('Please choose the supported kernel method. '
              'We use the default marten kernel to continue the program.')
        model = SingleTaskGP(dataset_x, dataset_f)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_model(mll)
    fit_gpytorch_mll(mll)
    return model, mll, mean, std


def next_point_bo(dataset, beta, bounds_low, kernel_type):
    dataset = dict([(key, dataset[key]) for key in ['x', 'f']])
    try:
        model, _, mean, std = fit_model_gp(dataset, kernel_type)
    except RuntimeError:
        print('Cannot fit the GP model, the result is undependable.')
        return False, None
    except botorch.exceptions.errors.ModelFittingError:
        print('Cannot fit the GP model, the result is undependable.')
        return False, None
    try:
        # next_y = upper_confidence_bound(bounds_low, model, beta)
        for i in range(5):
            next_x = upper_confidence_bound(
                bounds_low, model, beta + 0.2 * (i + 1))
            not_similar_count = 0
            for his_x in dataset['x']:
                temp = next_x - his_x
                # print(1, temp)
                # print(temp.abs().max())
                if temp.abs().max() >= 1e-3:
                    not_similar_count += 1
            # print(not_similar_count, dataset['x'].shape[0])
            if not_similar_count >= dataset['x'].shape[0] * 0.99:
                break
            else:
                print("next_x:",next_x)
                print("dataest:",dataset)
                print(
                    'Already have similar points, re-optimize the acquisition function.')
        # print(next_y)
    except gpytorch.utils.errors.NotPSDError:
        print('The optimization process have converged.')
        return False, None
    except RuntimeError:
        # error = 'RuntimeError'
        print('The optimization process have stopped early, the result is undependable.')
        return False, None
    return True, next_x, model, mean, std


def update_dataset_ucb(new_point_y, new_point_x, value, dataset):
    """
    add (new_point, value) to the normal dataset (NOT COMP DATASET)
    :param new_point_y: shape is (1, dim) or (dim)
    :param value: observed value of new point, shape is (1) or (1, 1)
    :param dataset: the dataset as {(x, f(x))}, which is a dict
    :return: the new dataset
    """
    if not dataset:
        return {'x': new_point_x, 'y': new_point_y, 'f': value}
    if len(value.shape) == 1:
        value = value.reshape(1, -1)

    if new_point_y != None and 'y' in dataset.keys():
        if len(new_point_y.shape) == 1:
            new_point_y = new_point_y.reshape(1, new_point_y.shape[-1])
        # if len(value.shape) == 1:
        #     value = value.reshape(1, 1)
        dataset['y'] = torch.cat([dataset['y'], new_point_y], 0)
        dataset['x'] = torch.cat([dataset['x'], new_point_x], 0)
        # print(1, dataset['f'], dataset['f'].shape)
        # print(2, value, value.shape)
        dataset['f'] = torch.cat([dataset['f'], value], 0)
    else:
        # if len(value.shape) == 1:
        #     value = value.reshape(1, 1)
        dataset['x'] = torch.cat([dataset['x'], new_point_x], 0)

        dataset['f'] = torch.cat([dataset['f'], value], 0)
    return dataset


def Baye(point,bounds, dim, func, kernel_type, node_dataset, itr,init_num=5):
    # bounds_nor = torch.tensor([[-1., 1.]] * dim).double()
    bounds_nor = torch.tensor([[0., 1.]] * dim,dtype = torch.float64)
    if (node_dataset is None or node_dataset['x'].shape[0] == 0):
        dataset = init_points_dataset_bo(init_num,bounds_nor, bounds, func)
        cost_budget = init_num
    elif node_dataset['x'].shape[0]<init_num:
        cost_budget = init_num - node_dataset['x'].shape[0]
        dataset = init_points_dataset_bo(cost_budget ,bounds_nor, bounds, func)
        # print(dataset)
        # print(node_dataset)
        #合并两个结果
        merged_dataset = {}
        for key, value in dataset.items():
            if key in node_dataset:
                merged_value = torch.cat((value, node_dataset[key]))
                merged_dataset[key] = merged_value
        dataset = merged_dataset

        # print("after update:",dataset)
    else:
        dataset = node_dataset
        cost_budget = 0
    
    #加入每个分支中点
    # init_x = (point - bounds.t()[0]) / (bounds.t()[1] - bounds.t()[0]) 
    # # init_x = (point - bounds.t()[0]) * 2 / (bounds.t()[1] - bounds.t()[0]) - 1
    # init_f = func(point).reshape(1, 1)
    
    # dataset = update_dataset_ucb(point, init_x, init_f, dataset)
    # print(dataset)
    iter_num = itr - cost_budget
    ucb_value = -np.inf
    for i in range(iter_num + 1):
        #beta越大，越是探索
        # print(i)
        _, next_x, model,dataset_mean , dataset_std = next_point_bo(dataset, 0.2 * dim * torch.log(torch.tensor(2 * dataset['f'].shape[0])) , bounds_nor, kernel_type)
        # print(next_x)
        if i == iter_num :
            # print(dataset)
            max_index = torch.argmax(dataset['f'])
            now_x = dataset['x'][max_index].unsqueeze(0)

            posterior = model.posterior(now_x)
            mean = posterior.mean.detach() 
            var = posterior.variance.detach()
            scale = math.floor(math.log10(abs((mean * dataset_std) + dataset_mean)))
            scale = math.pow(10,scale)
            
            now_ucb_value = (mean * dataset_std) + dataset_mean + scale * var
            
            posterior = model.posterior(next_x)
            mean = posterior.mean.detach() 
            var = posterior.variance.detach()
            
            scale = math.floor(math.log10(abs((mean * dataset_std) + dataset_mean)))
            scale = math.pow(10,scale)
            # scale = 10
            next_ucb_value = (mean * dataset_std) + dataset_mean + scale * var
            if next_ucb_value > now_ucb_value:
                ucb_value = next_ucb_value
            else:
                ucb_value = now_ucb_value
            print("ucb_value:",ucb_value)
        else:
            next_y = next_x * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
            # next_y = (next_x + 1) * (bounds.t()[1] - bounds.t()[0]) / 2 + bounds.t()[0]
            next_f = func(next_y).reshape(1, 1)
            dataset = update_dataset_ucb(next_y, next_x, next_f, dataset)
    
    
   
    max_index = torch.argmax(dataset['f'])
    now_x = dataset['x'][max_index].unsqueeze(0)
    # print(dataset)
    return dataset,dataset['f'].max(),ucb_value, now_x
    
