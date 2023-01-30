import multiprocessing
from multiprocessing.dummy import freeze_support
from os.path import exists
import os
from functools import partial
import networkx as nx
import numpy as np
from qiskit.optimization.applications.ising import max_cut

freeze_support()


def generate_graph(n: int, weight: str, graph_type: str) -> nx.Graph:
    """Generate random graph object
    given number of nodes, weight and graph type

    Args:
        n (int): number of nodes
        weight (str): type of weight
        graph_type (str): type of graph

    Returns:
        nx.Graph: the resulting graph
    """

    if graph_type == "cyclic":
        G = nx.cycle_graph(n)
    elif graph_type == "star":
        G = nx.star_graph(n - 1)
    elif graph_type == "erdos-renyi":
        G = nx.erdos_renyi_graph(n, p=0.5)  # NOTE: p = probability of connectivity
    elif graph_type == "barabasi-albert":
        G = nx.barabasi_albert_graph(n, m=2)  # NOTE: m = pore likhbo
    elif graph_type == "3regular":
        G = nx.random_regular_graph(3, n)

    if weight == "random":
        rand_weight = [np.random.randint(1, 10) for _ in range(G.number_of_edges())]
    elif weight == "none":
        rand_weight = [1 for _ in range(G.number_of_edges())]
    elif weight == "partial":
        rand_weight = [1 for _ in range(G.number_of_edges())]
        rand_weight[2] = 4  # np.random.randint(2, 10)
    elif weight == "unequal":
        rand_weight = [x + 1 for x in range(G.number_of_edges())]
    edge_list = []
    for i, g in enumerate(G.edges):
        g = list(g)
        g.append(rand_weight[i])
        edge_list.append(tuple(g))
    G.add_weighted_edges_from(edge_list)
    return G


def generate_objective(graph: nx.Graph) -> tuple:
    """Generate objective Hamiltonian matrix and summedop.

    Args:
        graph (nx.Graph): graph object to be converted

    Returns:
        tuple: objective hamiltonian summedop and matrix
    """
    n = graph.number_of_nodes()
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = graph.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
    qubit_op, _ = max_cut.get_operator(w)
    obj_hamiltonian = qubit_op.to_opflow()
    return obj_hamiltonian, obj_hamiltonian.to_spmatrix().diagonal()


def generating_and_saving(n, number_of_experiments):
    """Generate and save function for graph, objective
    hamiltonian matrix and summedop objects.

    Args:
        n (int): number of nodes for graph
        number_of_experiments (int): number of experiment to be generate
    """
    w = "random"
    graph_type = "barabasi-albert"
    print("weight type ", w)
    for sample_num in range(number_of_experiments + 1):
        print(f"Done with sample {sample_num} node {n}")
        G = generate_graph(n, w, graph_type)

        path = f"compare_data/graph/{graph_type}/"
        if not exists(path):
            os.makedirs(path)
        
        nx.write_gpickle(
            G,
            f"compare_data/graph/{graph_type}/{n}-nodes_{w}-weighted_exp{sample_num}.p",
        )
        obj_summedop, obj_matrix_diag = generate_objective(G)

        path = f"compare_data/obj/{graph_type}/"
        if not exists(path):
            os.makedirs(path)

        filename_summedop = f"compare_data/obj/{graph_type}/obj_summedop_nodes{n}_weight-{w}_exp{sample_num}"
        filename_matrix = f"compare_data/obj/{graph_type}/obj_matrix_nodes{n}_weight-{w}_exp{sample_num}"
        file_exists_summedop = exists(filename_summedop)
        # print(file_exists_summedop)
        if file_exists_summedop:
            print("the following file exists:")
            print(filename_summedop)
        else:
            print("saved")
            np.savez(filename_summedop, obj_summedop)
            np.savez(filename_matrix, obj_matrix_diag)


if __name__ == "__main__":

    total_experiments = 100
    nodes = range(6, 18 + 2, 2)
    pool = multiprocessing.Pool(20)
    generating_and_saving_x = partial(
        generating_and_saving, number_of_experiments=total_experiments
    )
    pool.map(generating_and_saving_x, nodes)

    print(f"COMPLETELY DONE FOR {nodes} NODES")
    pool.close()
