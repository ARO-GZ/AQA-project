from TNContraction.main import contraction_experiment
from GP.helper_functions import tn_to_graph, json_to_dict_list, bin_to_dec_partition
from TNContraction.circuits import build_toffoli_circuit
import json
import opt_einsum as oe

path_dict = {
    2: "TNContraction/results/toffoli_end_k2_pen0.023255813953488372.json",
    3: "TNContraction/results/toffoli_end_k3_pen0.023255813953488372.json",
    4: "TNContraction/results/toffoli_end_k4_pen0.023255813953488372.json",
    5: "TNContraction/results/toffoli_end_k5_pen0.023255813953488372.json",
    6: "TNContraction/results/toffoli_end_k6_pen0.023255813953488372.json",
    7: "TNContraction/results/toffoli_end_k7_pen0.021191860465116282.json",
    8:"TNContraction/results/toffoli_end_k8_pen0.018837209302325582.json"
}

k=3
result = json_to_dict_list(path_dict[k])

# print(result)
bits = result[0][0]["sol"]
print(bits.shape)


circ = build_toffoli_circuit()
G = tn_to_graph(circ.psi)

n = G.number_of_nodes()
print(n)
comm = bin_to_dec_partition(bits,n)

print(comm)

print(f"Running Contraction Experiment for k = {k}")

for c_order in ["naive", "ascending", "descending"]:

    print("--------------\n", c_order.upper(),"\n--------------\n" )

    # total_path, total_path_info, total_eq, flat_array_list = contraction_experiment(comm,G, community_order=c_order)

    total_path, total_path_info, greedy_path, greedy_path_info = contraction_experiment(comm,G, community_order=c_order, check_greedy=True)



    # print(dir(total_path_info))
    # exit()
    print("ANNEALING APPROACH\n----------")
    print("Speedup: ", total_path_info.speedup)
    print("Naive FLOPs", total_path_info.naive_cost)
    print("Optimized FLOPs: ", total_path_info.opt_cost)
    print("Largest Intermediate: ", total_path_info.largest_intermediate, "\n\n")
    # print(total_path_info)
    # exit()


    # greedy_path, greedy_path_info = oe.contract_path(total_eq, *flat_array_list, optimize="greedy")


    # print(total_path_info2)
    print("GREEDY APPROACH\n----------")
    print("Speedup: ", greedy_path_info.speedup)
    print("Naive: ", greedy_path_info.naive_cost)
    print("Optimized: ", greedy_path_info.opt_cost)
    print("Largest Intermediate: ", greedy_path_info.largest_intermediate, "\n\n")



