from typing import Union
import numpy as np
import opt_einsum as oe
import quimb as qu
import quimb.tensor as qtn
import networkx as nx
from collections import Counter

# einsum_chars = "abcdefghijklmnopqrstuvwxyz"
einsum_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'
# einsum_chars += einsum_chars.upper()

def get_edge_tups(network: Union[qtn.Circuit, nx.Graph]):
    """Return the edge tuples
    open legs -> (i,)
    closed legs -> (i,j)"""
    if type(network) == nx.Graph:
        return list(network.edges())
    else:
        return [tuple(edge) for edge in network.psi.ind_map.values()]

def get_edge_weights(network):
    '''
    network is a nx.Graph
    Returns: list with the edges weights in 
    same order you get the edges from network
    '''
    return [d['weight'] for (u,v,d) in network.edges(data=True)]

class Edge():
    def __init__(self, tup, index, hanging=False):
        self.n0, self.n1 = tup
        self.index = index
        self.letter = einsum_chars[index]
        self.hanging = hanging

    def __repr__(self):
        if self.hanging:
            return f"({self.n0}   {self.n1})"
        else:
            return f"({self.n0}---{self.n1})"

def build_edges(edge_tups):
    edges = []
    max_node = max([max(e) for e in edge_tups])
    h_i = 1
    for t_i, tup in enumerate(edge_tups):
        if len(tup) == 2:
            edges.append(Edge(tup, t_i))
        else:
            tup = (tup[0], max_node+h_i)
            edges.append(Edge(tup, t_i, True))
            h_i += 1
    
    return edges

def group_edges(edges, communities):
    """Groups edges based on a community list"""
    
    grouped_edges = [list() for _ in range(len(communities))]
    super_edges = []

    for edge in edges:
        internal  = False
        for c_i, community in enumerate(communities):
            if edge.n0 in community and (edge.hanging or edge.n1 in community):
                grouped_edges[c_i].append(edge)
                internal = True
                break

        if not internal:
            super_edges.append(edge)

    return grouped_edges, super_edges

# TODO
def get_arrays_from_network(network):
    pass

def get_all_subpaths(communities, grouped_edges, super_edges, sub_network_optimize, edge_weigths = None):
    paths = []
    # path_infos = []
    eqs = []
    all_arrays =[]
    for comm, edge_group in zip(communities, grouped_edges):
        path, path_info, eq, arrays = get_subpath(comm, edge_group, super_edges, sub_network_optimize)
        all_arrays.append(arrays)
        eqs.append(eq)
        paths.append(path)
        # path_infos.append(path_info)
    
    return paths, eqs, all_arrays

def get_full_equation(eqs):

    super_LHS, super_RHS = "", ""
    total_LHS, total_RHS = "", ""
    letter_counter = Counter()

    for eq in eqs:
        LHS, RHS = eq.split("->")
        letter_counter.update(RHS)
        super_LHS += RHS + ","
        total_LHS += LHS + ","

    super_LHS = super_LHS[:-1]
    total_LHS = total_LHS[:-1]

    for letter in letter_counter:
        if letter_counter[letter] == 1:
            super_RHS += letter
            total_RHS += letter

    super_eq = super_LHS + "->" + super_RHS
    total_eq = total_LHS + "->" + total_RHS

    return total_eq, super_eq

def get_total_path(super_eq, paths, communities):
    super_arrays = get_arrays_from_eq(super_eq)
    super_path, super_path_info = oe.contract_path(super_eq, *super_arrays, optimize = "greedy")

    corrected_paths = correct_subpaths(communities, paths)
    total_path = []
    for p in corrected_paths:
        total_path += p
    total_path += super_path

    return total_path

def get_custom_optimizer(total_path):

    class CustomOptimizer(oe.paths.PathOptimizer):
        def __init__(self, customized_path):
            self.customized_path = customized_path

        def __call__(self, inputs, output, size_dict, memory_limit=None):
            return self.customized_path

    custom_optim = CustomOptimizer(total_path)

    return custom_optim

def get_arrays_from_eq(eq):

    LHS, RHS = eq.split("->")
    LHS_arrs = LHS.split(",")

    arrays = []
    for s in LHS_arrs:
        shape = tuple([2 for _ in range(len(s))])
        arr = np.ones(shape)
        arrays.append(arr)

    return arrays

def get_einsum_equation(comm, edge_group, super_edges):
    arr_strings = {c:"" for c in comm}
    LHS, RHS = "",""
    for e in edge_group:
        arr_strings[e.n0] += e.letter
        if e.hanging:
            RHS += e.letter
        else:
            arr_strings[e.n1] += e.letter
    
    for e in super_edges:
        if e.n0 in comm:
            arr_strings[e.n0] += e.letter
            RHS += e.letter
        elif e.n1 in comm:
            arr_strings[e.n1] += e.letter
            RHS += e.letter
    
    for c in arr_strings:
        LHS += arr_strings[c] + ","
    
    eq = LHS[:-1] + "->" + RHS

    return eq

def get_subpath(comm, edge_group, super_edges, optimize):
    # arrays = get_arrays(edge_group, super_edges)
    eq = get_einsum_equation(comm, edge_group, super_edges)
    arrays = get_arrays_from_eq(eq)
    path, path_info = oe.contract_path(eq, *arrays, optimize = optimize)
    return path, path_info, eq, arrays

def correct_subpaths(communities, paths):

    corrected_paths = []
    comm_lens = [len(comm) for comm in communities]
    corrections = [sum(comm_lens[i+1:]) + i for i in range(len(comm_lens))]

    for p_i, p in enumerate(paths):
        N_arrs_shifted = 0
        corrected_path = []
        correction = corrections[p_i]
        for c_i, cont in enumerate(p):
            arr1_index, arr2_index = cont
            last_left_arr_index = len(p) - c_i - N_arrs_shifted

            shift1 = False
            if (arr1_index > last_left_arr_index):
                arr1_index += correction
                N_arrs_shifted -= 1

            
            shift2 = False
            if (arr2_index > last_left_arr_index):
                arr2_index += correction
                N_arrs_shifted -= 1
            
            N_arrs_shifted += 1
            
            corrected_path.append((arr1_index, arr2_index))           

        corrected_paths.append(corrected_path)
    
    return corrected_paths