import os, sys

root = os.getcwd()[:-11]
print(root)
sys.path.insert(0,root)
# exit()

from pickletools import optimize
from typing import Union, List
import quimb as qu
import quimb.tensor as qtn
import opt_einsum as oe
import networkx as nx
# from GP.graph_partitioning import GraphPartition
from TNContraction.cont_help_functions import (get_all_subpaths, get_edge_tups,
        build_edges, get_edge_weights, group_edges, get_full_equation, get_total_path, get_custom_optimizer)
NoneType = type(None)


def contraction_experiment(communities: List[List[int]], 
                            network: Union[qtn.Circuit, nx.Graph],
                            sub_network_optimize = "greedy",
                            edge_weight_as_dim = False,
                            community_order = "naive",
                            check_greedy = False
                            ):
    
    if community_order not in ["naive", "ascending", "descending"]:
        raise ValueError("community_order must be 'naive', 'ascending' or 'descending'")

    edge_tups = get_edge_tups(network)

    print(edge_tups)

    # exit()

    edges = build_edges(edge_tups)

    if edge_weight_as_dim:
        if isinstance(network, qtn.Circuit):
            raise ValueError("Can only use edge weights for nx.Graph")
        edge_weights = get_edge_weights(network)
    else:
        edge_weights = None

    

    if check_greedy:
        N_nodes = sum([len(comm) for comm in communities])
        single_community = list(range(N_nodes))
        greedy_paths, greedy_eqs, greedy_arrays = get_all_subpaths([single_community], [edges], [], "greedy", edge_weigths=edge_weights )
        # greedy_path = greedy_paths[0]
        greedy_eq = greedy_eqs[0]
        greedy_arrays = greedy_arrays[0]

        greedy_path, greedy_path_info = oe.contract_path(greedy_eq, *greedy_arrays, optimize="greedy")



    grouped_edges, super_edges = group_edges(edges, communities)
    paths, eqs, all_arrays = get_all_subpaths(communities, grouped_edges, super_edges, sub_network_optimize, edge_weigths = edge_weights)

    if community_order != "naive":
    # if False:
        eq_RHS_lens = [len(eq.split("->")[-1]) for eq in eqs]
        eq_i = list(enumerate(eq_RHS_lens))

        # print(eq_i)

        if community_order == "ascending":
            eq_i.sort(key=lambda tup: tup[1])
        elif community_order == "descending":
            eq_i.sort(key=lambda tup: tup[1], reverse=True)

        new_indices = [tup[0] for tup in eq_i]

        paths = [paths[i] for i in new_indices]
        eqs = [eqs[i] for i in new_indices]
        all_arrays = [all_arrays[i] for i in new_indices]
        communities = [communities[i] for i in new_indices]     

    total_eq, super_eq = get_full_equation(eqs)

    total_path = get_total_path(super_eq, paths, communities)

    custom_optimizer = get_custom_optimizer(total_path)

    flat_array_list = []
    for sublist in all_arrays:
        flat_array_list += sublist

    total_path, total_path_info = oe.contract_path(total_eq, *flat_array_list, optimize=custom_optimizer)

    if check_greedy:
        return total_path, total_path_info, greedy_path, greedy_path_info
    else:
        return total_path, total_path_info, total_eq, flat_array_list

if __name__ == "__main__":
    from Data.GraphData import graph_dict
    # G = graph_dict["Zachary"]()
    # params = {
    #     "communities": [
    #         [0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21], 
    #         [4, 5, 6, 10, 16], 
    #         [8, 9, 14, 15, 18, 20, 22, 26, 29, 30, 32, 33], 
    #         [23, 24, 25, 27, 28, 31]
    #     ],
    #     "network": G
    # }

    G = nx.frucht_graph()
    params = {
        "communities": [
            [0, 1, 2, 3, 4], 
            [5, 6, 7, 8, 9],
            [10, 11]
        ],
        "network": G
    }

    total_path, total_path_info, total_eq, flat_array_list = contraction_experiment(**params)
    print(total_path_info)
    total_path2, path_info2 = oe.contract_path(total_eq, *flat_array_list, optimize="greedy")
    print(path_info2)

    # communities: List[List[int]], 
    #                         network: Union[qtn.Circuit, nx.Graph],
    #                         sub_network_optimize = "greedy",
    #                         edge_weight_as_dim = False
