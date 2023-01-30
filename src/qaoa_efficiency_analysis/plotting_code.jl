using Pkg
Pkg.activate(".")

using PyPlot
using JLD2
using Optim
using LaTeXStrings

include("qaoa_optimizers.jl")
include("data_analysis.jl")

plot_colors = [
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

PyPlot.rc("text", usetex = true)
PyPlot.rc("font", family = "serif", size = 11)


function plot_qaoa_tqaoa_initialization()

    dir_out = "../compare_data/data/barabasi-albert/5_to_100_layers"    
    dir_out2 = "../compare_data/data/barabasi-albert/t-vs-qaoa"
    dir_out3 = "../compare_data/data/barabasi-albert/t-zero-rand"
                                    
    experiment_results = extract_info(dir_out);
    exp_info = get_experiment_info(dir_out)

    exp_res = extract_info(dir_out2,possibilities=[("qaoa", "rand-rand")])
    merge!(exp_res,extract_info(dir_out3,possibilities=[("t-qaoa", "rand-rand")]))
    merge!(exp_res,experiment_results)

    b_samples = get_best_samples(exp_res)

    methods = ["qaoa","t-qaoa"]
    x = 5:5:100
    
    fig, axs = PyPlot.subplots(1, 2, figsize = (6,2.7), constrained_layout = true)
    nodes = exp_info["nodes_list"]
    graph_type = exp_info["graph_type"]

    for (j,method) = enumerate(methods)
        method_poss = [("$method", "rand-rand"),("$method", "zero-rand")]
        labels = ["rand-rand","zero-rand"]

        for (pos,label) = zip(method_poss,labels)
            if method == "qaoa"
                clr ="#ff7f00"
                marker = "^"
                mfc ="none"
                if label == "rand-rand" 
                    clr = "#377eb8" 
                    marker = "p"
                end
            elseif method == "t-qaoa"
                clr = "#4daf4a"
                marker = "s"
                mfc ="none"
                if label == "rand-rand" 
                    clr = "#984ea3"
                    marker = "X"
                end
            end

            y_mean = [median(exp_res[(nodes, pos, k)]["energy"]) for k=x] 
            y_best = [b_samples[(nodes,pos, k)]["energy"] for k=x]
            y_quartile_down = [quantile(exp_res[(nodes, pos, k)]["energy"],0.25) for k=x]
            y_quartile_up = [quantile(exp_res[(nodes, pos, k)]["energy"],0.75) for k=x]

            axs[j].plot(x,y_mean,color = clr,label = label, marker =marker, mfc =mfc, markersize = 4 )
            axs[j].plot(x,y_best,color = clr, "--", marker=marker, mfc = mfc, markersize = 4)
            axs[j].fill_between(x,y_quartile_down,y_quartile_up, alpha=0.3,color = clr)
        end
        
        axs[j].legend(fontsize=8)
        axs[j].set_title(uppercase(method),fontsize=10)
        if j==1
            axs[j].set_ylabel(L"E_\mathrm{norm}", fontsize = 10)
        end
        axs[j].grid(linestyle="--" , linewidth = 0.6)
        if method == "t-qaoa"
            axs[j].set_yscale("log")
        end
    end
    fig.supxlabel("Number of layers", fontsize =10)
    fig.savefig("../plot/$graph_type/qaoa-t_qaoa-rr_vs_zr-energy-vs-layersb.pdf", bbox_inches="tight")

end

function all_methods_layers_nodes()
    dir_out = "../compare_data/data/barabasi-albert/5_to_100_layers/"                                    
    info_wanted = ["energy"]
    exp_info = get_experiment_info(dir_out);
    experiment_results = extract_info(dir_out,info_wanted = ["energy"],nodes_list = exp_info["nodes_list"]);
    best_sample = get_best_samples(experiment_results)

    labels = ["QAOA","T-QAOA","HOHo-QAOA"]
    possibilities = [("qaoa", "zero-rand"),("t-qaoa", "zero-rand"),("nc-qaoa", "zero-rand")]

    fig, axs = PyPlot.subplots(1, 2, figsize = (6,2.7),constrained_layout = true)

    x = 5:5:100
    nodes = exp_info["nodes_list"]

    for (pos,label) = zip(possibilities,labels)
        if pos[1] == "qaoa"
            clr = "#ff7f00"
            marker = "^"
        elseif pos[1] == "t-qaoa"
            clr = "#4daf4a"
            marker = "s"
        elseif pos[1] == "nc-qaoa"
            clr = "#f781bf"
            marker = "o"
        end
        y_mean = [median(experiment_results[(nodes, pos, k)]["energy"]) for k=x] 
        y_best = [best_sample[(nodes,pos, k)]["energy"] for k=x]
        y_quartile_down = [quantile(experiment_results[(nodes, pos, k)]["energy"],0.25) for k=x]
        y_quartile_up = [quantile(experiment_results[(nodes, pos, k)]["energy"],0.75) for k=x]
        axs[1].plot(x,y_mean,color = clr,label = label, marker = marker, markersize = 4, mfc = "none")
        axs[1].plot(x,y_best,color =clr, "--", marker = marker, markersize = 4, mfc = "none")
        axs[1].fill_between(x,y_quartile_down,y_quartile_up, alpha=0.3,color = clr)
    end

    axs[1].set_xlabel("Number of layers", fontsize = 10)
    axs[1].set_ylabel(L"E^*_\mathrm{norm}",fontsize = 10)
    axs[1].set_yscale("log")
    axs[1].grid(linestyle="--" , linewidth = 0.6)

    fig.legend(bbox_to_anchor=(0.55, 1.09),loc = "upper center",ncol = 3,borderaxespad=0.5, fontsize=8)

    dir_out = "../compare_data/data/barabasi-albert/qaoa-vs-t-vs-nc"
    nodes = 6:2:18
    info_wanted = ["energy"]
    exp_info = get_experiment_info(dir_out);
    experiment_results = extract_info(dir_out,info_wanted = ["energy"],nodes_list = nodes);
    best_sample = get_best_samples(experiment_results)

    graph_type = exp_info["graph_type"]
    kmax = exp_info["kmax"]

    for (pos,label) = zip(possibilities,labels)
        if pos[1] == "qaoa"
            clr = "#ff7f00"
            marker = "^"
        elseif pos[1] == "t-qaoa"
            clr = "#4daf4a"
            marker = "s"
        elseif pos[1] == "nc-qaoa"
            clr = "#f781bf"
            marker = "o"
        end
        y_mean = [median(experiment_results[(node, pos, kmax)]["energy"]) for node=nodes] 
        y_best = [best_sample[(node,pos, kmax)]["energy"] for node=nodes]
        y_quartile_down = [quantile(experiment_results[(node, pos, kmax)]["energy"],0.25) for node=nodes]
        y_quartile_up = [quantile(experiment_results[(node, pos, kmax)]["energy"],0.75) for node=nodes]
        axs[2].plot(nodes,y_mean,color = clr,label = label, marker = marker, markersize = 4, mfc = "none")
        axs[2].plot(nodes,y_best,color =clr, "--", marker = marker, markersize = 4, mfc = "none")
        axs[2].fill_between(nodes,y_quartile_down,y_quartile_up, alpha=0.3,color = clr)
    end

    axs[2].set_xlabel("Number of nodes", fontsize= 10)
    axs[2].set_yscale("log")
    axs[2].grid(linestyle="--" , linewidth = 0.6)

    PyPlot.savefig("../plot/$graph_type/methods-comparison-energy-vs-layers-nodes.pdf", bbox_inches="tight")
end

function main()
    plot_qaoa_tqaoa_initialization()
    all_methods_layers_nodes()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end