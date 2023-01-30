import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from qiskit import *
from qiskit.circuit import Parameter
from hamiltonian_generate import generate_graph, generate_objective
import pickle
from os.path import exists

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)

# mixer
def mixer(beta: float, n: int) -> QuantumCircuit:
    """Generates mixer hamiltonian qiskit circuit

    Args:
        beta (float): parameter for rotation in X gate in Mixer
        n (int): number of qubits

    Returns:
        QuantumCircuit: mixer circuit
    """
    mixer_qc = QuantumCircuit(n)
    for q in range(n):
        mixer_qc.rx(2 * beta, q)
    return mixer_qc


# objective
def objective(gamma: float, graph: nx.Graph) -> QuantumCircuit:
    """Generates mixer hamiltonian qiskit circuit

    Args:
        gamma (float): parameter for rotation in exp(-iZZ) gate for QAOA objective
        graph (nx.Graph): graph sample to build  objective hamiltonian

    Returns:
        QuantumCircuit: objective hamiltonian circuit
    """
    n = graph.number_of_nodes()
    obj_qc = QuantumCircuit(n)
    for edge in graph.edges.data("weight"):
        q1 = edge[0]
        q2 = edge[1]
        w = edge[2]
        obj_qc.cx([q1], [q2])
        obj_qc.rz(w * gamma, [q2])
        obj_qc.cx([q1], [q2])
    return obj_qc


# QAOA circuit
def qaoa(param_list: list, graph: nx.Graph) -> QuantumCircuit:
    """_summary_

    Args:
        param_list (list): _description_
        graph (nx.Graph): _description_

    Returns:
        QuantumCircuit: _description_
    """
    n = graph.number_of_nodes()
    qc_maxcut = QuantumCircuit(n)
    beta = param_list[: len(param_list) // 2]
    gamma = param_list[len(param_list) // 2 : len(param_list)]
    for q in range(n):
        qc_maxcut.h(q)
    for b in range(len(beta)):
        qc_maxcut = qc_maxcut.compose(objective(gamma[b], graph))
        qc_maxcut = qc_maxcut.compose(mixer(beta[b], n))
    return qc_maxcut


def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )


def data_generate_landscape_nonlinearity_2d_subplots_all_layer(
    layer_considered: int, layer: int, graph: nx.classes.graph.Graph, divider: int, simulator
    ):
    """
    TO GENERATE DATA FOR 2D PLOT TO SHOW ENERGY NONLINEARITY IN QAOA

    INPUT:
    layer_considered: Decides for how many layers you want to see the nonlinearity
    layer: The total number of layers in QAOA
    divider: Defines a number that divides "pi" by "divider" to create intermedicate points for dense plotting
    simulator: Qiskit simulator for "nodes < 12" one use 'statevector_simulator' and "nodes > 12" use "qasm_simulator"

    RETURN:
    A 2D plot with 2 subplots: LHS is variation of energy with objective parameters and RHS with mixer parameters

    """
    obj, obj_diag = generate_objective(graph)
    obj_matrix = obj.to_matrix()
    min_eig, max_eig = np.min(obj_diag).real, np.max(obj_diag).real
    param_list = [Parameter(f"x{i}") for i in range(2 * layer)]
    variable_ang_list = np.arange(0, np.pi + np.pi / divider, np.pi / divider)
    mixer_ang_list, obj_ang_list = [1.0], variable_ang_list


    for mixer_ang in mixer_ang_list:
        for layer_no in range(layer_considered):
            initial_angle = [1.0] * (2 * layer)
            
            filename = f"plot_data/nonlinearity/energy_with_obj_layer-{layer_no+1}_divider-{divider}.p"
            file_exists = exists(filename)
            
            if file_exists:
                print(f"the follwing file exists: {filename}")
            else:
                U = []
                for obj_ang in obj_ang_list:
                    initial_angle[layer_no], initial_angle[layer + layer_no] = (
                        mixer_ang,
                        obj_ang,
                    )
                    ansatz = qaoa(param_list, graph)
                    ansatz = ansatz.bind_parameters(initial_angle)
                    result = execute(ansatz, simulator).result()
                    state_vector = np.asmatrix(result.get_statevector())
                    energy = (state_vector @ obj_matrix @ state_vector.getH()).real
                    norm_en = (energy - min_eig) / (max_eig - min_eig)
                    U.append(float(norm_en))
                with open(filename, "wb") as fp:
                    pickle.dump(U, fp)
            

    mixer_ang_list, obj_ang_list = variable_ang_list, [1.0]
    for obj_ang in obj_ang_list:
        for layer_no in range(layer_considered):
            initial_angle = [1.0] * (2 * layer)
            filename = f"plot_data/nonlinearity/energy_with_mix_layer-{layer_no+1}_divider-{divider}.p"
            file_exists = exists(filename)

            if file_exists:
                print(f"the follwing file exists: {filename}")

            else:
                U = []
                for mixer_ang in mixer_ang_list:
                    initial_angle[layer_no], initial_angle[layer + layer_no] = (
                        mixer_ang,
                        obj_ang,
                    )
                    ansatz = qaoa(param_list, graph)
                    ansatz = ansatz.bind_parameters(initial_angle)
                    result = execute(ansatz, simulator).result()
                    state_vector = np.asmatrix(result.get_statevector())
                    energy = (state_vector @ obj_matrix @ state_vector.getH()).real
                    norm_en = (energy - min_eig) / (max_eig - min_eig)
                    U.append(float(norm_en))
                with open(filename, "wb") as fp:
                    pickle.dump(U, fp)
            
            
            
            


def plot_landscape_nonlinearity_2d_subplots_all_layer(
    layer_considered: int, layer: int, divider: int
):
    """


    INPUT:
    layer_considered: Decides for how many layers you want to see the nonlinearity
    layer: The total number of layers in QAOA
    divider: Defines a number that divides "pi" by "divider" to create intermedicate points for dense plotting

    RETURN:
    A 2D plot with 2 subplots: LHS is variation of energy with objective parameters and RHS with mixer parameters

    """
    variable_ang_list = np.arange(0, np.pi + np.pi / divider, np.pi / divider)
    obj_ang_list = variable_ang_list
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
    line_style = ["r-", "b--", "g-."]

    for layer_no in range(layer_considered):
        with open(f"plot_data/nonlinearity/energy_with_obj_layer-{layer_no+1}_divider-{divider}.p", "rb") as fp:
            en = pickle.load(fp)  
        ax1.plot(
            obj_ang_list,
            en,
            line_style[layer_no],
            mfc="none",
        )
        ax1.plot(obj_ang_list, en, line_style[layer_no], mfc="none")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 20))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_xlabel("$\\gamma_j$", fontsize = 10)
    ax1.set_ylabel("$E_{\\small\\textrm{norm}}$", fontsize = 10)
    mixer_ang_list= variable_ang_list
    
    for layer_no in range(layer_considered):
        with open(f"plot_data/nonlinearity/energy_with_mix_layer-{layer_no+1}_divider-{divider}.p", "rb") as fp:
            en = pickle.load(fp)
        if layer_no +1 == 1:
            pos = 'st'
        elif layer_no +1 == 2:
            pos = 'nd'
        elif layer_no +1 == 3:
            pos = 'rd'
        else:
            pos = 'th'
        
        ax2.plot(
            mixer_ang_list,
            en,
            line_style[layer_no],
            label= f"{layer_no+1}{ pos} layer",
            mfc="none",
        )
    
    ax2.set_xlabel("$\\beta_j$", fontsize = 10)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 20))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    fig.legend(ncol=layer, bbox_to_anchor=(0.7, 0.97), fontsize=8)
    # ax2.legend(loc='upper center', ncol = layer, bbox_to_anchor=[0.5, 1.095], borderaxespad = 1.5, fontsize = 8)#, frameon=False)
    fig.savefig(
        f"plot/energy_landscape_2d_subplots_all-layer_total_layer-{layer}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        f"plot/energy_landscape_2d_subplots_all-layer_total_layer-{layer}.png",
        bbox_inches="tight",
    )



def landscape_nonlinearity_3d(layer: int, graph: nx.classes.graph.Graph, simulator):
    """
    INPUT:
    layer: The total number of layers in QAOA

    RETURN:
    A 3D plot for variation of energy with QAOA parameters

    """

    obj, obj_diag = generate_objective(graph)
    obj_matrix = obj.to_matrix()
    min_eig, max_eig = np.min(obj_diag).real, np.max(obj_diag).real
    param_list = [Parameter(f"x{i}") for i in range(2 * layer)]
    variable_ang_list = np.arange(0, np.pi + np.pi / 150, np.pi / 150)

    mixer_ang_list, obj_ang_list = variable_ang_list, variable_ang_list
    sx, sy = mixer_ang_list.size, obj_ang_list.size
    theta_oracle_list_plot, theta_x_list_plot = (
        np.tile(mixer_ang_list, (sy, 1)),
        np.tile(obj_ang_list, (sx, 1)).T,
    )
    energy_landscape_list = np.zeros((len(mixer_ang_list), len(obj_ang_list)))
    for no, mixer_ang in enumerate(mixer_ang_list):
        U = []
        for obj_ang in obj_ang_list:
            ansatz = qaoa(param_list, graph)
            other_layer = np.random.random(2 * (layer - 1))
            initial_angle = [mixer_ang, obj_ang]
            if len(other_layer) != 0:
                initial_angle = initial_angle + list(other_layer)
            ansatz = ansatz.bind_parameters(initial_angle)
            result = execute(ansatz, simulator).result()
            state_vector = np.asmatrix(result.get_statevector())
            energy = (state_vector @ obj_matrix @ state_vector.getH()).real
            norm_en = (energy - min_eig) / (max_eig - min_eig)
            U.append(norm_en)
        energy_landscape_list[no] = U

    _, ax1 = plt.subplots(
        1, 1, figsize=(10, 4.6), subplot_kw=dict(projection="3d")
    )
    norm = plt.Normalize(mixer_ang_list.min(), obj_ang_list.max())
    colors = cm.viridis(norm(theta_oracle_list_plot))
    rcount, ccount, _ = colors.shape
    surf = ax1.plot_surface(
        theta_oracle_list_plot,
        theta_x_list_plot,
        energy_landscape_list,
        rcount=rcount,
        ccount=ccount,
        facecolors=colors,
        shade=False,
    )
    surf.set_facecolor((0, 0, 0, 0))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(2 * np.pi))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_xlabel("$\\beta_j^{p=1}$", labelpad=10)
    ax1.set_ylabel("$\\gamma_j^{p=1}$", labelpad=10)
    ax1.set_zlabel("$E_{\\small\\textrm{norm}}$", labelpad=10)
    ax1.zaxis.set_label_position("bottom")
    plt.savefig(f"plot/energy_landscape_3d_layer_{layer}.pdf")
    plt.savefig(f"plot/energy_landscape_3d_layer_{layer}.png")


if __name__ == "__main__":

    weight = ["none", "random", "partial", "unequal", "real"]
    graph_type = "barabasi-albert"
    weight = weight[1]
    layer = 3  # layers = layers of QAOA
    qubit = 10  # qubit = number of nodes/qubits

    if qubit < 12:
        simulator = Aer.get_backend("statevector_simulator")
    else:
        simulator = Aer.get_backend("qasm_simulator")
    graph = generate_graph(qubit, weight, graph_type)
    layer_considered = 3
    divider = 150
    plot_type_list = ["2D", "3D"]
    plot_type = plot_type_list[0]  # or 1

    if plot_type == "2D":
        data_generate_landscape_nonlinearity_2d_subplots_all_layer(layer_considered, layer, graph, divider, simulator)
        plot_landscape_nonlinearity_2d_subplots_all_layer(
            layer_considered, layer, divider
        )
    elif plot_type == "3D":
        landscape_nonlinearity_3d(layer, graph)
