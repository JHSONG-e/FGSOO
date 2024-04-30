# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
# import gym
import json
import os
import random
# import ObjFunc
#import imageio
import torch
from functions import Func


class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("inf")
        self.foldername = foldername
        print(foldername)
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        trace_path = self.foldername + '/result' + str(len( self.results) )
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')
            
    def track(self, result):
        if result < self.curt_best:
            self.curt_best = result
        self.results.append(self.curt_best)
        if len(self.results) % 100 == 0:
            self.dump_trace()





class Cassini2Gtopx:
    def __init__(self, fold_name, init, dims=22):
        self.dims = dims
        self.lb = np.array([-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
                            1.05, 1.05, 1.15, 1.7, -np.pi, -np.pi, -np.pi, -np.pi])
        self.ub = np.array([0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
                            6.5, 291.0, np.pi, np.pi, np.pi, np.pi])
        self.counter = 0
        self.tracker = tracker(fold_name)

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 10
        self.leaf_size = 10
        self.ninits = init
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        ObjFunc_x = torch.from_numpy(x)  # 把sample转为tensor类型
        ObjFunc_x = ObjFunc_x.reshape(1, self.dims)
        result = -Func.cassini2_gtopx(ObjFunc_x)
        result = np.float64(result)
        self.tracker.track(result)

        return result


    
    
    
    
    
    
    
    
    
    
    
    
    
