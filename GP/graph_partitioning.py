import numpy as np
import scipy.sparse as scp_sp
from GP.helper_functions import Z_mats, Z_mats_sp, group_solution, bin_to_dec_partition
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import neal 
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities as gmc
from networkx.algorithms.community import naive_greedy_modularity_communities as ngmc
from networkx.algorithms.community import modularity


class GraphPartition():
	"""
	Class for performing graph partitioning
	
	Atributes
	---------
		G : nx.Graph
			Graph to partition
		k : int
			Number of clusters
		is_sparse : bool
			Whether the graph is sparse or not
		M : np.array
			Modularity matrix for the QUBO
		S : np.array
			Constraint matrix for the QUBO

	Methods
	-------
		constraint()
			Builds the constraint matrix for the QUBO
		sparse_constraint()
			Builds a sparse version of the constraint matrix for the QUBO
		mod_mat()
			Builds the modularity matrix for the QUBO
		sparse_mod_mat()
			Builds a sparse version of the modularity matrix for the QUBO
		get_QUBO(pen)
			Builds the QUBO matrix for the given penalty
		quantum_solve(pen)
			Solves the QUBO using the D-Wave quantum annealer
		thermal_solve(pen)
			Solves the QUBO using the simulated annealing algorithm
		hybrid_solve(pen)
			Solves the QUBO using the hybrid quantum-classical annealer
		get_modularity(x)
			Computes the modularity given the partition x
	"""
	def __init__(self, G:nx.Graph=None, k:int = None, is_sparse:bool =True):

		if G == None or k== None:
			raise ValueError('Please provide a graph and number of communities')
	
		self.is_sparse = is_sparse
		self.G = G
		assert (2 <= k <= self.G.number_of_nodes()) and isinstance(k, int), "k must be between 2 and the number of nodes in the graph"
		self.k = k
		self.A = nx.adjacency_matrix(self.G)
		self.M2 = self.get_M2()
		self.B = self.get_B_matrix()
		self.S = self.get_constraint_matrix()

		# if self.is_sparse:
		# 	self.S = self.sparse_constraint()
		# 	self.M = self.sparse_mod_mat()
		# else:
		# 	self.S = self.constraint()
		# 	self.M = self.mod_mat()

	# def constraint(self):
	# 	n = self.G.number_of_nodes()

	# 	if self.k == 2:
	# 		return np.zeros((n,n))
		
	# 	# Contraint
	# 	Z = Z_mats(n,self.k)
	# 	S = np.zeros((n*self.k,n*self.k))
	# 	ones = np.ones((n*self.k,n*self.k))

	# 	for i in range(n):
	# 		S += Z[i]@ones@Z[i]-2*Z[i]
	# 	return S
	
	def get_M2(self):
		A_rows = scp_sp.csr_matrix.sum(self.A,0).reshape(-1,1) # 1xn matrix
		M2 = A_rows.sum()
		return M2
	

	def get_constraint_matrix(self):
		n = self.G.number_of_nodes()
		if self.k == 2:
			return scp_sp.csr_matrix(np.zeros((n,n)))
		# Contraint
		Z = Z_mats_sp(n,self.k)
		S = scp_sp.csr_matrix((n*self.k,n*self.k))
		ones = scp_sp.csr_matrix(np.ones((n*self.k,n*self.k)))

		for i in range(n):
			S += Z[i]*ones*Z[i]-2*Z[i]

		return S

	def get_B_matrix(self):
		
		A_rows = scp_sp.csr_matrix.sum(self.A,0).reshape(-1,1) # 1xn matrix
		g_i_g_j = A_rows@A_rows.T/self.M2
		B = self.A - g_i_g_j
		return B

	def get_QUBO(self,pen):

		if self.k == 2:
			return scp_sp.csr_matrix(self.get_M() + pen*self.S)
	
		else:
			B_coo = scp_sp.coo_matrix(self.B)
			return scp_sp.block_diag([B_coo]*self.k) + pen*self.S
	
	def get_M(self):
		if self.k == 2:
			return self.B
		else:
			B_coo = scp_sp.coo_matrix(self.B)
			return scp_sp.block_diag([B_coo]*self.k)
		

	def quantum_solve(self,pen):
		sampler = EmbeddingComposite(DWaveSampler())
		QUBO = self.get_QUBO(pen)
		sample_set = sampler.sample_qubo(QUBO, num_reads=1500,annealing_time=100, return_embedding=True)
		return sample_set

	def thermal_solve(self,pen):
		sa = neal.SimulatedAnnealingSampler()
		QUBO = self.get_QUBO(pen)
		sampleset = sa.sample_qubo(QUBO, num_reads=500)
		return sampleset.aggregate()

	def hybrid_solve(self,pen):
		sa = LeapHybridSampler()
		QUBO = self.get_QUBO(pen)
		sampleset = sa.sample_qubo(QUBO)
		return sampleset	

	def get_modularity(self,x):
		mod = float(x@(self.get_M())@x)*2/self.M2
		if k == 2:
			return mod
		else:
			return mod/2
		


class ClassicalGraphPartition():
	"""
	Classical graph partitioning using the Louvain algorithm
	Can be used to compare the results of the quantum annealer to the classical algorithm

	Parameters
	----------
		G : nx.Graph
			Graph to partition
		k : int
			Number of clusters

	Methods
	-------
		partition()
			Partitions the graph using the Louvain algorithm
	"""	
	
	def __init__(self, G:nx.Graph, k_min:int, k_max:int, gamma=1):
		self.G = G
		self.k_min = k_min
		self.k_max = k_max
		self.gamma = gamma
		self.results = {}

	def naive_greedy_solve(self):
		c = ngmc(self.G, resolution=self.gamma)
		mod = self.eval_mod(c)
		self.results['Naive Greedy'] = self.generate_res_dict(c, mod)
    
	def greedy_solve(self):
		c = gmc(self.G, resolution=self.gamma, cutoff = self.k_min, best_n = self.k_max)
		mod = self.eval_mod(c)
		self.results['Greedy'] = self.generate_res_dict(c, mod)

	def generate_res_dict(self, communities, mod):
		return {
			"communities": communities,
			"modularity": mod
		}

	def eval_mod(self, communities):
		mod = modularity(self.G, communities)
		return mod

if __name__ == "__main__":
	pass

	