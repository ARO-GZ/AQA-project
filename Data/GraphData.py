import networkx as nx
from scipy.io import mmread, loadmat
import numpy as np

def getDolphins():
    a = mmread('Data/PartitionPaper/soc-dolphins.mtx')
    G = nx.from_scipy_sparse_matrix(a)
    return G

def getPoliticalBooks():
    with open("Data/PartitionPaper/out.dimacs10-polbooks", "r") as f:
        lines = f.readlines()[1:]
    edge_tups = [tuple(int(s) for s in line.split()) for line in lines]
    G = nx.Graph(edge_tups)
    return G

def getJazz():
    with open("Data/PartitionPaper/out.arenas-jazz", "r") as f:
        lines = f.readlines()[1:]
    edge_tups = [tuple(int(s) for s in line.split()) for line in lines]
    G = nx.Graph(edge_tups)
    return G

def getElegans():
    with open("Data/PartitionPaper/out.arenas-meta", "r") as f:
        lines = f.readlines()[1:]
    edge_tups = [tuple(int(s) for s in line.split()) for line in lines]
    G = nx.Graph(edge_tups)
    return G

def getWindmill(n=4, k=2):
    G = nx.windmill_graph(n,k)
    return G

def getFrucht():
    G = nx.frucht_graph()
    return G

def getKrack():
    G = nx.krackhardt_kite_graph()
    return G

def getFlorentine():
    G = nx.florentine_families_graph()
    return G


graph_dict = {
    "Zachary": nx.karate_club_graph,
    "Dolphins": getDolphins,
    "LesMiserables": nx.les_miserables_graph,
    "PolBooks": getPoliticalBooks,
    "Jazz": getJazz,
    "Elegans": getElegans,
    "Windmill": getWindmill,
    "Frucht": getFrucht,
    "Krack": getKrack,
    "Florentine": getFlorentine
}

if __name__ == "__main__":
    f = graph_dict["Dolphins"]
    G = f()
    print(G)
