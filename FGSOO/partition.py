from PyXAB.partition.Node import P_node
from PyXAB.partition.Partition import Partition
import numpy as np
import copy
import pdb
import torch

class KaryPartition(Partition):
    """
    Implementation of K-ary Partition especially when K >= 3, i.e., Ternary, Quaternary, and so on
    """

    def __init__(self, domain=None, K=3, node=P_node):
        """
        Initialization of the K-ary Partition

        Parameters
        ----------
        domain: list(list)
            The domain of the objective function to be optimized, should be in the form of list of lists (hypercubes),
            i.e., [[range1], [range2], ... [range_d]], where [range_i] is a list indicating the domain's projection on
            the i-th dimension, e.g., [-1, 1]
        K: int
            The number of children of each parent, with the default choice to be 3
        node
            The node used in the partition, with the default choice to be P_node.
        """
        if domain is None:
            raise ValueError("domain is not provided to the K-ary Partition")
        self.K = K
        super(KaryPartition, self).__init__(domain=domain, node=node)

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard K-ary partition, i.e., split every
        parent node into K children nodes of the same size. If there are multiple dimensions, the dimension to split the
        parent is chosen randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        parent_domain = parent.get_domain()
        # print(parent_domain)
        # 原先是随机找到一维度
        # dim = np.random.randint(0, len(parent_domain))
        # selected_dim = parent_domain[dim]
        dataset = parent.data.copy()
        my_array = np.array(parent_domain)
        ranges = np.ptp(my_array, axis=1)
        dim = np.argmax(ranges)
        selected_dim = parent_domain[dim]
        new_nodes = []
        boundary_points = np.linspace(selected_dim[0], selected_dim[1], num=self.K + 1)
        for i in range(self.K):
            domain = copy.deepcopy(parent_domain)
            # print(domain)
            domain[dim] = [boundary_points[i],boundary_points[i + 1]]
            select_indices = (dataset['y'][:, dim] >= boundary_points[i]) & (dataset['y'][:, dim] <= boundary_points[i + 1])


            new_dataset = {
                'x':dataset['x'][select_indices],
                'y':dataset['y'][select_indices],
                'f':dataset['f'][select_indices]
            }
            new_dataset['x'][:,dim] = (new_dataset['y'][:,dim] - boundary_points[i])/(boundary_points[i+1]- boundary_points[i])
            # print(domain)
            node = self.node(
                depth=parent.get_depth() + 1,
                index=self.K * parent.get_index() - (self.K - i - 1),
                parent=parent,
                domain=domain,
                data=new_dataset
            )
            new_nodes.append(node)

        parent.update_children(new_nodes)

        if newlayer:
            self.node_list.append(new_nodes)
            self.depth += 1
        else:
            # print("depth" + str(parent.get_depth()))
            self.node_list[parent.get_depth() + 1] += new_nodes
    

