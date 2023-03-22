# Tensor Network Contraction with D-Wave Quantum Annealing

## Authors
- [@hjaleta](https://www.github.com/hjaleta)
- [@ARO-GZ](https://github.com/ARO-GZ)

## Description of the project
Repository for Project associated to Applied Quantum Algorithms ( WI4650 ) at TU Delft 2021/2022

This project utilizes D-wave's QPU and Hybrid Solvers to solve optimization problems related to Tensor Networks.

 The first topic at hand is *Graph Partitioning*. In this field of math one studies how to split a big graph into smaller clusters. A commonly used algorithm falls back on the concept of modularity which is maximum given an optimal partition of a graph. In reference [1] they propose a method for running modularity maximization algorithm on DWave Quantum Annealers. All the modules needed to solve a graph partitioning problem with this approach can be found in GP folder. 

 Tensor Networks are very useful for approximating large hyperspaces and quantum systems. The larger they get, the more difficult it is to find an efficient way of contracting them. In [2], Johnnie Gray and Stefanos Kourtis analyse different approaches to obtain the optimal contraction path. On of the methods consists on partitioning the graph representation of the corresponding Tensor Network. We merged this idea with our previous work so that we tried to optimize the contraction path of Tensor Networks using *Quantum Graph Partitioning*.

## A few words of caution 
Remember to install the requirements under `requirements.txt` before trying to run our code. 

In order for the imports to work correctly you must add the root folder of the repository to your python path.

If you want to send a job to DWave, you should also create an account and set up your API.


## References
[1] [Detecting Multiple Communities Using Quantum Annealing on the D-Wave System](https://arxiv.org/abs/1901.09756)

[2] [Hyper-optimized tensor network contraction](https://arxiv.org/abs/2002.01935)



