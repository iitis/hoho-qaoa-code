import multiprocessing
from multiprocessing.dummy import freeze_support
from os.path import exists
import os
import numpy as np
from qiskit.aqua.operators import PauliOp, SummedOp
from qiskit.quantum_info.operators import Pauli
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh


def get_obj(nodes, graph_type, weight, sample_num):
    """Generate objective hamiltonian

    Args:
        nodes (int): number of nodes
        graph_type (str): name of graph type
        weight (str): name of weight type
        sample_num (int): graph sample number

    Returns:
        (spmatrix): objective hamiltonian as spmatrix
    """
    obj_hamiltonian = np.load(
        f"compare_data/obj/{graph_type}/obj_summedop_nodes{nodes}_weight-{weight}_exp{sample_num}.npz",
        allow_pickle=True,
    )["arr_0"]
    return SummedOp(obj_hamiltonian).to_spmatrix()


def get_mixer(nodes):
    """Generate mixer Hamiltonian

    Args:
        nodes (int): number of nodes

    Returns:
        (spmatrix): mixer hamiltonian as spmatrix
    """
    mix_pauli_list = []
    for position in range(nodes):
        next_term = Pauli(
            ("I" * (nodes))[:position] + "X" + ("I" * (nodes))[position + 1 :]
        )
        mix_pauli_list.append(-1 * PauliOp(next_term))
    return SummedOp(mix_pauli_list).to_spmatrix()


freeze_support()

if __name__ == "__main__":

    nodes = 10
    energy_eval_for = ["step_parameter", "init_alpha"]
    energy_eval_for = [energy_eval_for[0]]
    sample_nums = range(1, 100 + 1)
    weight = "random"

    graph_type = "barabasi-albert"

    path = f"compare_data/data/{graph_type}/"
    if not exists(path):
        os.makedirs(path)

    for data_type in energy_eval_for:
        if data_type == "init_alpha":
            final_dict = {}
            for sample_num in sample_nums:
                print("-----------")
                print(sample_num)
                print("-----------")
                obj_hamiltonian_sp = csc_matrix(
                    get_obj(nodes, graph_type, weight, sample_num)
                )
                mix_hamiltonian_sp = csc_matrix(get_mixer(nodes))
                step_parameter_fix = 0.01
                init_alpha = 0.0
                true_dict_en = {"minima": [], "maxima": []}
                alpha_range = np.arange(
                    init_alpha, 1.0 + 0.0001, step_parameter_fix
                )
        
                filename = f"compare_data/data/{graph_type}/true_energy_with_init_qubit{nodes}-exp{sample_num}-initalpha0-step{step_parameter_fix}-weight{weight}.npy"
                file_exists = exists(filename)

                if file_exists:
                    print("the follwing file exists:")
                    print(filename)

                else:

                    def minmaxenergy(a):
                        """Calculate minimum and maximum from total hamiltonian eigevalues

                        Args:
                            a (float): alpha value

                        Returns:
                            min_en (float): minimum energy value
                            max_en (float): maximum energy value
                        """
                        print(a)

                        ham_sp = (
                            a * obj_hamiltonian_sp
                            + (1 - a) * mix_hamiltonian_sp
                        )

                        eigsv_small = eigsh(
                            ham_sp, 1, which="SA", return_eigenvectors=False
                        )[0].real
                        eigsv_big = eigsh(
                            ham_sp, 1, which="LA", return_eigenvectors=False
                        )[0].real

                        min_en = np.min(eigsv_small)
                        max_en = np.max(eigsv_big)
                        print(min_en, max_en)
                        print("---------------")
                        return min_en, max_en

                    pool = multiprocessing.Pool(3)
                    result = pool.map(minmaxenergy, alpha_range)
                    print(result)
                    true_dict_en["maxima"] = [x[1] for x in result]
                    true_dict_en["minima"] = [x[0] for x in result]
                    final_dict["energy"] = true_dict_en

                    np.save(filename, final_dict)
                    pool.close()

        elif data_type == "step_parameter":

            init_alpha_fix = 0.0
            final_dict = {}
            for sample_num in sample_nums:
                print("-----------")
                print(sample_num)
                print("-----------")
                obj_hamiltonian_sp = csc_matrix(
                    get_obj(nodes, graph_type, weight, sample_num)
                )
                mix_hamiltonian_sp = csc_matrix(get_mixer(nodes))
                for step_parameter in [
                    1.0,
                    1 / 2,
                    1 / 4,
                    1 / 6,
                    1 / 10,
                    1 / 20,
                    1 / 40,
                    1 / 60,
                    1 / 80,
                    1 / 100,
                    1 / 200,
                    1 / 400,
                    1 / 600,
                    1 / 1000,
                    1 / 2000,
                    1 / 6000,
                    1 / 10000,
                ]:
                    print(step_parameter)
                    print("-----------")
                    true_dict_en = {"minima": [], "maxima": []}
                    alpha_range = [1]
                    filename = f"compare_data/data/{graph_type}/true_energy_with_step_qubit{nodes}-exp{sample_num}-initalpha{init_alpha_fix}-step{step_parameter}-weight{weight}.npy"
                    file_exists = exists(filename)

                    if file_exists:
                        print("the follwing file exists:")
                        print(filename)

                    else:

                        def minmaxenergy(a):
                            """Calculate minimum and maximum from total hamiltonian eigevalues

                            Args:
                                a (float): alpha value

                            Returns:
                                min_en (float): minimum energy value
                                max_en (float): maximum energy value
                            """
                            print(a)
                            ham_sp = (
                                a * obj_hamiltonian_sp
                                + (1 - a) * mix_hamiltonian_sp
                            )
                            eigsv_small = eigsh(
                                ham_sp, 1, which="SA", return_eigenvectors=False
                            )[0].real
                            eigsv_big = eigsh(
                                ham_sp, 1, which="LA", return_eigenvectors=False
                            )[0].real

                            min_en = np.min(eigsv_small)
                            max_en = np.max(eigsv_big)
                            return min_en, max_en

                        pool = multiprocessing.Pool(2)
                        result = pool.map(minmaxenergy, alpha_range)
                        print(result)
                        true_dict_en["maxima"] = [x[1] for x in result]
                        true_dict_en["minima"] = [x[0] for x in result]
                        final_dict["energy"] = true_dict_en

                        np.save(filename, final_dict)
                        pool.close()

        else:
            print("Ai input chini na go")
            exit()
