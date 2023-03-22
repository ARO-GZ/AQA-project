import numpy as np
import scipy.sparse as scp_sp
from GP.helper_functions import Z_mats, Z_mats_sp, group_solution
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
	def __init__(self, G:nx.Graph=None, k:int = None, is_sparse:bool =False):

		if G == None or k== None:
			raise ValueError('Please provide a graph and number of communities')
	
		self.is_sparse = is_sparse
		self.G = G
		self.k = k
		self.M = self.mod_mat()

		if self.is_sparse:
			self.S = self.sparse_constraint()
			self.M = self.sparse_mod_mat()
		else:
			self.S = self.constraint()
			self.M = self.mod_mat()

	def constraint(self):
		n = self.G.number_of_nodes()
		# Contraint
		Z = Z_mats(n,self.k)
		S = np.zeros((n*self.k,n*self.k))
		ones = np.ones((n*self.k,n*self.k))

		for i in range(n):
			S += Z[i]@ones@Z[i]-2*Z[i]
		return S
	
	def sparse_constraint(self):
		n = self.G.number_of_nodes()
		# Contraint
		Z = Z_mats_sp(n,self.k)
		S = scp_sp.csr_matrix((n*self.k,n*self.k))
		ones = scp_sp.csr_matrix(np.ones((n*self.k,n*self.k)))

		for i in range(n):
			S += Z[i]*ones*Z[i]-2*Z[i]

		return S

	def mod_mat(self):
		n = self.G.number_of_nodes()
		M = np.zeros((n*self.k,n*self.k))
		A = nx.adjacency_matrix(self.G)
		A_rows = scp_sp.csr_matrix.sum(A,0) # 1xn matrix
		M2 = scp_sp.csr_matrix.sum(A)

		for i in range(n):
			for j in range(n):
				for m in range(self.k):
					M[i+m*n,j+m*n] = 1/(M2)*(A[i,j]-(A_rows[0,i]*A_rows[0,j])/M2)
		
		return M
	
	def sparse_mod_mat(self):
		n = self.G.number_of_nodes()
		M = np.zeros((n,n))
		A = nx.adjacency_matrix(self.G)
		A_rows = scp_sp.csr_matrix.sum(A,0) # 1xn matrix
		M2 = scp_sp.csr_matrix.sum(A)

		for i in range(n):
			for j in range(n):
				M[i,j] = 1/(M2)*(A[i,j]-(A_rows[0,i]*A_rows[0,j])/M2)
		
		return scp_sp.block_diag(self.k*[M])

	def get_QUBO(self,pen):
		if self.is_sparse:
			QUBO = (-self.M+pen*self.S).todense() 
		else:
			QUBO = -self.M+pen*self.S 
		return QUBO

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
		return x@(self.M)@x


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
