import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 10}
)

def grafitti_type(init_type):
    """Generates string for latex formatting amoung parameters initialization options

    Args:
        init_type (str): initialization type name

    Returns:
        grafitti (str): string formatted for latex usage
    """

    if init_type == 'rand-rand':
        grafitti = '$\\gamma_j^{\\tiny\\textrm{init}}\\sim \\textrm{U}(0, 2\\pi), \\beta_j^{\\tiny\\textrm{init}}\\sim \\textrm{U}(0, 2\\pi)$'
        grafitti = '$ \\textrm{RR} $'
    elif init_type == 'nearzero-rand':
        grafitti = '$\\gamma_j^{\\tiny\\textrm{init}}\\sim \\textrm{U}(0, 0.05), \\beta_j^{\\tiny\\textrm{init}}\\sim \\textrm{U}(0, 2\\pi)$'
        grafitti = '$ \\textrm{NZR} $'
    elif init_type == 'zero-rand':
        grafitti = '$\\gamma_j^{\\tiny\\textrm{init}}=0, \\beta_j^{\\tiny\\textrm{init}}\\sim \\textrm{U}(0, 2\\pi)$'
        grafitti = '$ \\textrm{ZR} $'
    return grafitti


class Variables:
    nodes_list = [6, 10, 16]
    nodes = 10
    layer = 10
    alpha_step_list_large = sorted(
        [
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
        ]
    )
    alpha_step_list_small = sorted(
        [
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
        ]
    )
    no_experiments = range(1, 100 + 1)
    x_ref = np.arange(0.0, 1 + 0.01, 0.001)

    init_alpha_list = np.arange(0.0, 1 + 0.001, 0.05)
    all_initialization = ["rand-rand", "nearzero-rand", "zero-rand"]
    graph_type = "barabasi-albert"


######################### different init_alpha with optimal energy plot #########################

def optimal_en_init_alpha_plot(nodes, init_alpha_list, initialization, graph_type, experiments_list):
    """Load data from energy values and returns lists for plotting

    Args:
        nodes [list(str)]: list of number of nodes
        init_alpha_list [list(float)]: list initial alpha values
        initialization [list(str)]: list with parameters initization names
        graph_type (str): graph type name
        experiments_list [list(int)]: list of objective hamiltonian indices

    Returns:
        y_axis [list(float)] : list of energy normalized
        x_axis [list(float)] : list of initial alpha values
        up_fill [list(float)] : list of energy normalized + standart deviation
        down_fill [list(float)] : list of energy normalized - standart deviation

    """
    for _, init_type in enumerate(initialization):
        y_axis, x_axis, up_fill, down_fill = [], [], [], []
        for _, init_alpha in enumerate(init_alpha_list):
            init_alpha = round(init_alpha, 5)
            mc_qaoa_norm_optimal_en = np.load(f'plot_data/{graph_type}/qubit{nodes}_norm_optimal_energy_init_alpha-{init_alpha}_{init_type}_tot-exp-{len(experiments_list)}.npy')    
            std =  np.load(f'plot_data/{graph_type}/qubit{nodes}_optimal_std_init_alpha-{init_alpha}_{init_type}_tot-exp-{len(experiments_list)}.npy')
            up_fill.append(mc_qaoa_norm_optimal_en + std)
            down_fill.append(mc_qaoa_norm_optimal_en - std)
            y_axis.append(mc_qaoa_norm_optimal_en)
            x_axis.append(init_alpha)


    return y_axis, x_axis, up_fill, down_fill


def optimal_en_alpha_step_plot(nodes, alpha_step_list, initialization, graph_type, experiments_list):
    """Load data from energy values and returns lists for plotting

    Args:
        nodes [list(str)]: list of number of nodes
        init_alpha_list [list(float)]: list of alpha step values
        initialization [list(str)]: list with parameters initization names
        graph_type (str): graph type name
        experiments_list [list(int)]: list of objective hamiltonian indices

    Returns:
        y_axis [list(float)] : list of energy normalized
        x_axis [list(float)] : list of alpha step values
        up_fill [list(float)] : list of energy normalized + standart deviation
        down_fill [list(float)] : list of energy normalized - standart deviation

    """
    for _, init_type in enumerate(initialization):
        y_axis, x_axis, up_fill, down_fill = [], [], [], []
        for _, alpha_step in enumerate(alpha_step_list):
            mc_qaoa_norm_optimal_en = np.load(f'plot_data/{graph_type}/qubit{nodes}_layer{Variables.layer}_norm_optimal_energy_alpha_step-{alpha_step}_{init_type}_tot-exp-{len(experiments_list)}.npy')    
            std =  np.load(f'plot_data/{graph_type}/qubit{nodes}_layer{Variables.layer}_optimal_std_alpha_step-{alpha_step}_{init_type}_tot-exp-{len(experiments_list)}.npy')
            up_fill.append(mc_qaoa_norm_optimal_en + std)
            down_fill.append(mc_qaoa_norm_optimal_en - std)
            y_axis.append(mc_qaoa_norm_optimal_en)
            x_axis.append(alpha_step)
    return y_axis, x_axis, up_fill, down_fill


def all_initialization():
    """Make plots for parameters initialization energy comparison"""
    layer = 3
    init_alpha_list = np.arange( 0.0, 0.5+0.001, 0.05)
    assert layer == 3, 'All initialization is resource consuming so keep it layer = 3'
    _, ax = plt.subplots( 1, 3, sharey=True, figsize = (6, 3) )
    def get_cmap(w):
        return plt.cm.get_cmap('coolwarm', w)
    cmap = get_cmap(len(init_alpha_list))

    for init_type_no, init_type in enumerate(Variables.all_initialization):
        grafitti = grafitti_type(init_type)  
        for no, init_alpha in enumerate(init_alpha_list):
            init_alpha = round(init_alpha, 8)
            x_axis = np.arange( init_alpha, 1+0.001, 0.01)
            mc_qaoa_norm_optimal_en = np.load(f'plot_data/{Variables.graph_type}/qubit{Variables.nodes}_layer{layer}_norm_energy_init_alpha-{init_alpha}_{init_type}_tot-exp-{len(Variables.no_experiments)}.npy')
            ax[init_type_no].plot(x_axis, mc_qaoa_norm_optimal_en, c = cmap(no))
        ax[init_type_no].set_title( f"{grafitti}" , fontsize = 10)
        if init_type_no == 1:
            ax[init_type_no].set_xlabel( '$\\alpha$', fontsize = 10)
        if init_type_no == 0:
            ax[init_type_no].set_ylabel( '$E^*_{\\small\\textrm{norm}}$',fontsize = 10 )
        ax[init_type_no].plot( np.arange( 0.0, 1+0.001, 0.01), [0]*len( np.arange( 0.0, 1+0.001, 0.01)), 'k--')
        ax[init_type_no].grid( True, which="both", linestyle='--', linewidth = 0.6)
    plt.tight_layout()
    plt.savefig(f'plot/{Variables.graph_type}/init_comparison.pdf')
    plt.savefig(f'plot/{Variables.graph_type}/init_comparison.png')

def alpha_step_init_superimposed():
    """Make plots for parameters initialization energy comparison with alpha step"""
    marking = [ 'v', '.', 'x' ]
    markingsize = [3.5, 8, 4]
    cool_colors = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.9))#, constrained_layout=True)
    for no, nodes in enumerate(Variables.nodes_list):
        y_axis, x_axis, up_fill, down_fill = optimal_en_init_alpha_plot(nodes, Variables.init_alpha_list, ['zero-rand'], Variables.graph_type, Variables.no_experiments)
        ax1.plot(x_axis, y_axis, label = f'{nodes} qubits', color=cool_colors[no], marker = marking[no], markersize = markingsize[no], linewidth = 1.5, mfc='none')
        ax1.fill_between(x_axis, down_fill, up_fill, alpha = 0.3)
        ax1.grid(True, which= 'both', linestyle='--', linewidth = 0.6)
    fig.legend(fontsize = 8, ncol = 3, loc = 9, bbox_to_anchor=(0.5,0.975))
    ax1.set_xlabel( '$\\alpha_{\\small\\textrm{init}}$', fontsize = 10)
    ax1.set_ylabel( '$E^{*}_{\\small\\textrm{norm}}(\\alpha=1)$', fontsize = 10 )
    ax1.text(0.01,0.35, '$(a)$')
    for no, nodes in enumerate(Variables.nodes_list):
        if nodes != 16:
            y_axis,x_axis, up_fill, down_fill = optimal_en_alpha_step_plot(nodes, Variables.alpha_step_list_large, ['zero-rand'], Variables.graph_type, Variables.no_experiments)
            ax2.loglog(x_axis, y_axis, label = f'{nodes} qubits', color=cool_colors[no], marker = marking[no],markersize = markingsize[no], linewidth = 1.5, mfc='none')
            ax2.fill_between(x_axis, down_fill, up_fill, alpha=0.3)
        else:
            y_axis,x_axis, up_fill, down_fill = optimal_en_alpha_step_plot(nodes, Variables.alpha_step_list_small, ['zero-rand'], Variables.graph_type, Variables.no_experiments)
            ax2.loglog(x_axis, y_axis, label = f'{nodes} qubits', color=cool_colors[no], marker = marking[no], markersize = markingsize[no], linewidth = 1.5, mfc='none')
            ax2.fill_between(x_axis, down_fill, up_fill, alpha = 0.3)
        ax2.grid( True, which="both", linestyle='--', linewidth = 0.6)
    ax2.set_xlabel( '$\\alpha_{\\small\\textrm{step}}$', fontsize = 10 )
    ax2.set_xticks([1e-4, 1e-2, max(x_axis)])
    ax2.text(0.0001,0.3, '$(b)$')
    plt.savefig(f'plot/{Variables.graph_type}/superimposed.pdf', bbox_inches='tight')
    plt.savefig(f'plot/{Variables.graph_type}/superimposed.png', bbox_inches='tight')

if __name__ == "__main__":

    # alpha_step_init_superimposed()
    all_initialization()
