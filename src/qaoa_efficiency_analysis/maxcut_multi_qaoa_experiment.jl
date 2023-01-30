using Dates
using Distributed
using Pkg
Pkg.activate(".")


# ----------------- Experiment parameters ----------------------------
threads_no = 19
addprocs(threads_no)

no_experiments = 100
possibilities = [("qaoa", "zero-rand"), ("t-qaoa", "zero-rand"), ("nc-qaoa", "zero-rand")]
nodes_list = 10

init_alpha_fix = 0.0
alpha_step_fix = 0.01

parallel = true

weight = "random"
graph_type = "barabasi-albert"
kmin = 4
kmax = 5:5:100

maxiter = 10000

optwith = "optim"
solver = "lbfgs"
period = false
level_repeat = 4

exp_tests_init = Iterators.product(
    1:no_experiments,
    init_alpha_fix,
    possibilities,
    alpha_step_fix,
    nodes_list,
    kmax
)
# _____________________________________


@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Distributions
##
@everywhere using JLD2
@everywhere using FileIO
@everywhere using NPZ
@everywhere include("qaoa_optimizers.jl")

experiment_info = Dict(
    "threads_no" => threads_no,
    "no_experiments" => no_experiments,
    "possibilities" => possibilities,
    "nodes_list" => nodes_list,
    "init_alpha_fix" => init_alpha_fix,
    "alpha_step_fix" => alpha_step_fix,
    "parallel" => parallel,
    "weight" => weight,
    "graph_type" => graph_type,
    "nodes_list" => nodes_list,
    "kmax" => kmax,
    "kmin" => kmin,
    "maxiter" => maxiter,
    "optwith" => optwith,
    "solver" => solver,
    "period" => period,
    "level_repeat" => level_repeat,
)



if length(ARGS) != 0
    dir_out = ARGS[findfirst(x -> x == "-outnew", ARGS)+1]
else
    dir_out = "../compare_data/data/$graph_type/data_$(Dates.now())"
    mkdir(dir_out)
end
save("$dir_out/experiment_info.jld2", experiment_info)

for run = 1:1 # HACK
    function generate_experiment(
        i::Int,
        init_alpha::Number,
        pos,
        alpha_step::Float64,
        nodes::Int,
        kmax::Int,
    )
        # println()
        # println("-------------------------")
        @show nodes, i, init_alpha, pos, alpha_step, nodes, kmax
        # println("-------------------------")
        # println()

        method, init_type = pos

        if method in ["qaoa", "nc-qaoa"]
            k = kmax
        else
            method == "tnc-qaoa"
            k = kmin
        end

        if init_type == "rand-rand" # QAOA and T-QAOA
            init_times = vcat(rand(Uniform(0, pi), k), rand(Uniform(0, pi), k))
        elseif init_type == "zero-rand" #All methods
            init_times = vcat(zeros(k), rand(Uniform(0, pi), k))
        elseif init_type == "rand-zero" #All methods
            init_times = vcat(rand(Uniform(0, pi), k), zeros(k))
        else
            throw(ArgumentError("incorrect init_type $init_type"))
        end

        hamiltonian =
            real.(
                npzread(
                    "../compare_data/obj/$graph_type/obj_matrix_nodes$(nodes)_weight-$(weight)_exp$i.npz",
                )["arr_0"]
            )
        results = Dict()

        n = Int(log2(length(hamiltonian)))
        d = load_sparsers(n)

        filename = "$dir_out/$method-$init_type-nodes$(nodes)-weight-$weight-exp$i-layers-$(kmax)--result.jld2"
        if isfile(filename)
            println("file $filename exists")
            return true
        end

        if method == "qaoa"
            results["result"] = qaoa(
                hamiltonian,
                kmax,
                d,
                alpha=1.0,
                optwith=optwith,
                solver=solver,
                maxiter=length(init_alpha:alpha_step:1) * maxiter,
                periodic=period,
                init_times=init_times,
            )
        elseif method == "nc-qaoa"
            results["result"] = nc_qaoa(
                hamiltonian,
                kmax,
                d,
                init_alpha=init_alpha,
                alpha_step=alpha_step,
                solver=solver,
                init_times=init_times,
            )
        elseif method == "t-qaoa"
            results["result"] = qaoa_trajectories_periodic(
                hamiltonian,
                kmax,
                kmin=k,
                d=d,
                level_repeat=level_repeat,
                solver=solver,
                init_angles=init_times,
            )
        else
            @assert false "where is the method? you gave me $(method)"
        end

        results["init_times"] = init_times
        save(filename, results)
        println("Done for $filename")
        true
    end
    (parallel ? pmap : map)(i -> generate_experiment(i...), exp_tests_init)
end

rmprocs()
