import numpy as np
import networkx as nx
import scipy.sparse as scp_sp
import json
colors = ['red','blue','green','yellow','purple','orange','black','white','strawberry_blond','blue_as_hjalmars_eyes']

def Z_mats(n,k):
	Z = []
	for i in range(n):
		A = np.zeros((n*k,n*k))
		for j in range(k):
			A[i+j*n,i+j*n] = 1
		Z.append(A) 
	return Z

def Z_mats_sp(n,k):
	Z = []
	index0 = n*(np.array([i for i in range(k+1)]))[1:]-n
	A = np.ones((k))	
	for i in range(n):
		Z.append(scp_sp.coo_matrix((A,(index0+i,index0+i)),shape = (n*k,n*k))) 
	return Z


def group_solution(sampleset,n,k):
	di = {}
	for array in sampleset.record:
		arr = str(array[0])+str(',')+str(array[1])
		if arr in di.keys():
			di[arr] += 1
		else:
			di[arr] = 1
	return di

def check_validity(sampleset, n):

    bit_lists = [record[0] for record in sampleset.record]
    bit_lists = [bit_list.reshape(-1,n) for bit_list in bit_lists]
    node_sums_list = [np.sum(bit_list, axis = 0) for bit_list in bit_lists]
    
    node_validities = []
    for node_sums in node_sums_list:
        invalid_nodes = 0
        for node_sum in node_sums:
            if node_sum != 1:
                invalid_nodes += 1
        node_validities.append(invalid_nodes)

    return node_validities

def plot_partition(GP,x):
	col_map = []
	n = GP.G.number_of_nodes()
	k = GP.k
	for i in range(n):
		for j in range(k):
			if x[i+n*j]==1:
				col_map.append(colors[j])
				break
	nx.draw(GP.G, node_color = col_map)

def plot_solution(G,x,k):
	col_map = []
	node_list = []
	n = G.number_of_nodes()

	for j in range(k):
		col_map = col_map+len(x[j])*[colors[j]]
		node_list = node_list+x[j]

	nx.draw(G, nodelist = node_list, node_color = col_map)

def rec_to_dict_list(record, GP):
	
	record_dicts = []
	for rec in record:
		d = {}
		x = rec[0]
		d["sol"] = x
		d["freq"] = rec[2]
		d["H_con"] = x@GP.S@x
		d["mod"] = x@GP.M@x 
		record_dicts.append(d)
	
	record_dicts.sort(key=lambda d: d["H_con"])
	
	split_indices = []
	for r_i in range(len(record_dicts)-1):
		if record_dicts[r_i]["H_con"] != record_dicts[r_i+1]["H_con"]:
			split_indices.append(r_i+1)
	
	sub_lists = []
	for s0, s1 in zip([0] + split_indices, split_indices + [len(record_dicts)]):
		sub_list = record_dicts[s0:s1]
		sub_lists.append(sub_list)
	
	# sorted_results = []
	for sub_list in sub_lists:
		sub_list.sort(key=lambda d: d["mod"], reverse=True)
		# sorted_results += sub_list
	
	# return sorted_results
	return sub_lists



def dict_list_to_json(path, dict_list):
	if path[-5:] != ".json":
		path += ".json"
	
	json_data = []

	for sub_list in dict_list:
		sub_list_json = []
		for d in sub_list:
			d_json = {}
			d_json["sol"] = [int(b) for b in d["sol"]]
			d_json["freq"] = int(d["freq"])
			d_json["H_con"] = int(d["H_con"])
			d_json["mod"] = float(d["mod"])
			sub_list_json.append(d_json)
		json_data.append(sub_list_json)
	
	with open(path, 'w') as outfile:
		json.dump(json_data, outfile)
	
def json_to_dict_list(path):

	with open(path, 'r') as f:
		dict_list = json.load(f)
	
	for sub_list in dict_list:
		for d in sub_list:
			d["sol"] = np.array(d["sol"])
	
	return dict_list

def tn_to_graph(tn):
	'''
	Converts a quimb TensorNetwork representing a Quantum Circuit
	to a Networkx graph.

	NOTE: Each tensor is a node and each bond an edge. 
	The QC will be generally of the form U|psi0>, thus the outer 
	layer of tensors (gates) will have open links. This ones are not
	included here, since to do so we would need to introduce "empty"
	nodes. However, this approach will suffice as we are interested 
	in finding the optimal contraction path for the existing TN

	Args:
		tn: a quimb TensorNetwork instance

	Returns:
		G: 	 networkx graph
	'''

	G = nx.Graph()

	#number of nodes/tensors
	N = len(tn.tensors)

	G.add_nodes_from(range(N))

	edges = tn.ind_map.values()

	# list of tuples representing edges from the indices map of tn
	edges_list = [tuple(edge) for edge in tn.ind_map.values() if len(edge)==2]

	G.add_edges_from(edges_list)

	return G

def bin_to_dec_partition(arr, n):

	assert isinstance(arr, np.ndarray)
	arr = arr.reshape(-1, n)
	k = arr.shape[0]
	partition = [list() for _ in range(k)]
	for (k_i, n_i), bit in np.ndenumerate(arr):
		if bit == 1:
			partition[k_i].append(n_i)

	return partition

def array_to_dict(sol,S,M):
    d = {}
    d["sol"] = sol
    d["freq"] = 1
    d["H_con"] = (sol@S@sol.T)
    d["mod"] = (sol@M@sol.T)
    return d

def fix_multiple_results(GP,sols,n_sols,check:str='random'):
    sorted_sols = [subsublist for sublist in sols for subsublist in sublist]
    sorted_sols.sort(key=lambda d: d["mod"],reverse=True)
    pop_size = int(GP.G.number_of_nodes()/2.5)

    if len(sorted_sols)>n_sols:
        best_n_sols = sorted_sols[0:(n_sols+1)]
    else:
        best_n_sols = sorted_sols
    fixed_sols = []
	
    for sol in best_n_sols:
        fixed_sol = fix_results(GP,sol,check=check,pop_size=pop_size)
        fixed_sols.append(array_to_dict(fixed_sol,GP.S,GP.M))
    fixed_sols.sort(key=lambda d: d["mod"],reverse=True)

    return fixed_sols  

def fix_results(gp, sol, check = "random", pop_size = 10, n_best = 1):
	"""
	Correct a QUBO solution

	If check = 'random' and pop_size is bigger than the total number of possible combinations the function
	will switch mode to check = 'all'

	Args:
		gp: GraphPartition
			An instance of a GraphPartition class object corresponding to a graph
		sol: dict
			An associated solution dictionary 
		check: str
			string indicating how to search for solutions. Could be 'random' or 'all'
		pop_size: int
			How many possible solutions to check. Only relevant if check == 'random'
		n_best: int
			How many of the best modularity solutions to return
		n_best: int
			How many of the worst modularity solutions to return
	"""

	n = gp.G.number_of_nodes()
	if sol["H_con"] == -n:
		return sol['sol']

	x = sol["sol"].reshape(-1,n)
	k = x.shape[0]

	node_activations = np.sum(x, axis = 0)
	invalid_nodes = np.argwhere(node_activations != 1).flatten()
	N_inv = len(invalid_nodes)
	n_combos = k**N_inv

	new_x = x.copy()
	new_x[:,invalid_nodes] = 0

	if check == "all":
		pop_size = n_combos
	elif check == "random" and pop_size > n_combos:
		check = "all"
		pop_size = n_combos

	if n_best>pop_size:
		raise ValueError(f"'n_best' + 'n_worst' is bigger than population size")

	pop_x = np.tile(new_x,(pop_size, 1, 1) )

	if check == "random":        
		new_i = np.random.randint(0,k, size = (pop_size, N_inv))

	elif check == "all":
		indices = [np.arange(k) for _ in range(N_inv)]
		new_i = np.array(np.meshgrid(*indices)).T.reshape(-1, N_inv)

	pop_range = np.arange(pop_size)
	for j in range(N_inv):
		pop_x[pop_range,new_i[:,j], invalid_nodes[j]] = 1

	mod_mat = gp.M
	pop_x = pop_x.reshape((pop_size,-1))
	# xQ = pop_x@mod_mat
	# mods = np.sum( xQ * pop_x, axis = 1)
	mods = np.diag(pop_x@mod_mat@pop_x.T)

	sorted_indices = np.flip(np.argsort(mods))

	assert n_best >0
	# best_sols = pop_x[sorted_indices[:n_best],:]
	best_sols = pop_x[sorted_indices,:]
	return best_sols[0]
	#return [best_sols[i] for i in range(len(best_sols))] 

if __name__ == "__main__":
	from GP.graph_partitioning import GraphPartition
	from Data.GraphData import graph_dict
	from TNContraction.detect_commuties import optimize_pen
	G = graph_dict["LesMiserables"]()
	# path = "GP/results/Elegance/eleg_pen_0dot00265_H_invalid.json"
	# sol = json_to_dict_list(path)
	# print(sol
	# GP = GraphPartition(G=G, k=5, sparse=True)

	sol , pen = optimize_pen(G,6,n_fix=3,hybrid=True,pen=0.017,narrow=True,check='random')
	path = 'Draft/results/LesMis/Miserables_pen_H'+str(pen)

	dict_list_to_json(path,[[sol]])
	# G = graph_dict["Elegan"]()

	# print(GP.S.shape)
	# new_sol = fix_results(GP, sol[0][0])

	# print(new_sol.shape)
	# print(new_sol@GP.S@new_sol)

	# print(GP.get_modularity(new_sol))


