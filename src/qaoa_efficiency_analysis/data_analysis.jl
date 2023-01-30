# using Pkg
# Pkg.activate(".")

using JLD2
using Optim
using DataStructures
using NPZ

"""
    get_experiment_info(directory::String)

Loads the dictionary saved in `experiment_info.jdl2` from a given directory path.
If ``experiment_info.jdl2`` does not exists, it will raise an assertion error. 

"""
function get_experiment_info(directory::String)
    file_exp_name = "$directory/experiment_info.jld2"
    @assert isfile(file_exp_name) "$file_exp_name doesn't exist, and it should"

    return load(file_exp_name)
end


"""
    extract_info(dir_out, info_wanted = ["energy"], possibilities = nothing, no_experiments= nothing, nodes_list = nothing, kmax = nothing, save_results = false)   


Extract results from experiment files in a directory and return as a dictionary.
Give a path to the experiment folder (`dir_out`) and information type (`info_wanted = ["energy,"iterations","angles"]`).


The function has 5 keywords:
* `possibilities`, which experiment setups you want select. Ex: `[("qaoa", "rand-rand"),("t-qaoa", "zero-rand")`.
* `no_experiments`, number of hamiltonians in the experiments. 
* `nodes_list`, list of nodes for the hamiltonians.
* `kmax`, maximum number of layers. 
* `save_results`, save the results at `dir_out` path.

"""
function extract_info(
    dir_out::String;
    info_wanted = ["energy"],
    possibilities = nothing,
    no_experiments= nothing,
    nodes_list = nothing,
    kmax = nothing,
    save_results = false,
)   
    experiment_info = get_experiment_info(dir_out)

    weight = experiment_info["weight"]
    graph_type = experiment_info["graph_type"]

    if possibilities == nothing
        possibilities = experiment_info["possibilities"]
    end
    if no_experiments == nothing
        no_experiments= experiment_info["no_experiments"]
    end
    if nodes_list == nothing
        nodes_list = experiment_info["nodes_list"]
    end
    if kmax == nothing
        kmax = experiment_info["kmax"]
    end

    experiment_results = Dict()
    for nodes in nodes_list, pos in possibilities, info in info_wanted, k in kmax
        experiment_results[(nodes, pos, k)] =
            Dict(info_wanted .=> [[] for x in lastindex(info_wanted)])
        method, init_type = pos
        for i = 1:no_experiments
            hamiltonian =
                real.(
                    npzread(
                        "../compare_data/obj/$graph_type/obj_matrix_nodes$(nodes)_weight-$(weight)_exp$i.npz",
                    )["arr_0"]
                )
            filename = "$dir_out/$method-$init_type-nodes$(nodes)-weight-$weight-exp$i--result.jld2"
            if typeof(kmax) != Int
                filename = "$dir_out/$method-$init_type-nodes$(nodes)-weight-$weight-exp$i-layers-$(k)--result.jld2"
            end
            if info == "energy"
                if method == "qaoa"
                    rs = load(filename)["result"]
                elseif method == "tnc-qaoa"
                    rs = load(filename)["result"][end][end]
                else
                    rs = load(filename)["result"][end]
                end
                energy = Optim.minimum(rs)
                norm_energy =
                    (energy - minimum(hamiltonian)) /
                    (maximum(hamiltonian) - minimum(hamiltonian))
                push!(experiment_results[(nodes, pos, k)]["energy"], norm_energy)
            end
            if info == "iterations"
                if method == "qaoa"
                    iterations = Optim.iterations(load(filename)["result"])
                elseif method == "tnc-qaoa"
                    rs =
                        Optim.iterations.(
                            collect(Iterators.flatten(load(filename)["result"]))
                        )
                    iterations = sum(rs)
                else
                    iterations = sum(Optim.iterations.(load(filename)["result"]))
                end
                push!(experiment_results[(nodes, pos, k)]["iterations"], iterations)
            end
            if info == "angles"
                if method == "qaoa"
                    rs = load(filename)["result"]
                elseif method == "tnc-qaoa"
                    rs = load(filename)["result"][end][end]
                else
                    rs = load(filename)["result"][end]
                end
                angles = Optim.minimizer(rs)
                push!(experiment_results[(nodes, pos, k)]["angles"], angles)
            end
        end
    end
    if save_results == true
        return save("$dir_out/experiment_results.jld2", "info", experiment_results)
    end
    return experiment_results
end

"""
    get_best_samples(experiment_results, number_of_samples)

Returns a dictionary with best sample from `experiment_results` dictionary.
You can set, optionally, a number of samples of choice.
The keys of `best_samples` are the same of the `experiment_results`.

"""
function get_best_samples(experiment_results::Dict; samples_num::Int=1)
    best_samples = Dict()
    for key in keys(experiment_results)
        best_samples[key] = Dict()
        data = experiment_results[key]["energy"]
        samples_ids = [findall(x -> x == i, data)[1] for i in nsmallest(samples_num, data)]
        best_samples[key]["samples_ids"] = samples_ids
        for key2 in keys(experiment_results[key])
            best_samples[key][key2] = [experiment_results[key][key2][i] for i in samples_ids
]
        end
    end
    return best_samples
end


"""
    get_period_objective(hamiltonian)

Returns period value (float) for a given objective hamiltonian.

"""
function get_period_objective(hamiltonian::Vector{ComplexF64})
    hamiltonian = real.(hamiltonian)
    hamiltonian = hamiltonian - minimum(hamiltonian) * ones(length(hamiltonian))
    @assert norm(round.(hamiltonian, digits = 10) - hamiltonian) <= 1e-10
    t_dash = 2 / (reduce(gcd, round.(Int, hamiltonian)))
    return t_dash * pi
end


