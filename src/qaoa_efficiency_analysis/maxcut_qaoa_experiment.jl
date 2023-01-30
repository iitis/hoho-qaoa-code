using Distributed
using Pkg
Pkg.activate(".")

# multiprocessing parameters
threads_no = 4
addprocs(threads_no)

# parallelized parameters
no_experiments = 10
solver_list = ["lbfgs"]
possibilities = [[ false, false, "optim" ]] # FORMAT: ['period', 'bounds', 'software'] addon: [ true, false, "optim" ], [ false, false, "scipy" ], [ false, true, "scipy" ]
initialization = ["nearzero-rand", "rand-rand" ]
init_alpha_list = 0:0.05:1.0
alpha_step_list = [1/2000, 1/6000, 1/10000]


init_alpha_fix = 0.0
alpha_step_fix = 0.01

exp_tests_init = Iterators.product(1:no_experiments, init_alpha_list, solver_list, possibilities, initialization, alpha_step_fix)
exp_tests_step = Iterators.product(1:no_experiments, init_alpha_fix, solver_list, possibilities, initialization, alpha_step_list)

parallel = true

# common parameters for experiments
weight = "random"
graph_type = "barabasi-albert"
nodes = 10

k = 10
maxiter = 10000

# how many procs

@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Distributions
##
@everywhere using JLD2
@everywhere using FileIO
@everywhere using NPZ
@everywhere include("qaoa_optimizers.jl")
##


for run in 1:1 # HACK
    function generate_experiment(i::Int, init_alpha:: Number, solver::String, pos, init_type::String, alpha_step:: Number)

        @show i, init_alpha, solver, pos, init_type, alpha_step

        period, bound, optwith = pos
        if init_type == "rand-rand"
            init_times = vcat(rand(Uniform(0, 2*pi), k),rand(Uniform(0, 2*pi), k))
        elseif init_type == "zero-rand"
            init_times = vcat(zeros(k),rand(Uniform(0, 2*pi), k))
        elseif init_type == "nearzero-rand"
            init_times = vcat(rand(Uniform(0, 0.05), k),rand(Uniform(0, 2*pi), k))
        else
            throw(ArgumentError("incorrect init_type $init_type"))
        end

        hamiltonian = real.(npzread("../compare_data/obj/$(graph_type)/obj_matrix_nodes$(nodes)_weight-$(weight)_exp$i.npz")["arr_0"])
        results_qaoa = Dict()
        results_mcqaoa = Dict()

        n = Int(log2(length(hamiltonian)))
        d = load_sparsers(n)

        filename_mcqaoa = "../compare_data/$(graph_type)/exp$i-result_mc_qaoa_with_qubit$n-layer$k-initalpha-$init_alpha-inittyp-$init_type-alpha_step$alpha_step-exp$i-software$optwith-solver$solver-weight$weight-periodic$period-bounds$bound.jld2"

        # if isfile(filename_mcqaoa)
        #     println("file $filename_mcqaoa exists")
        #     return true
        # end

        # original QAOA - run he
        results_qaoa["result"] = qaoa(hamiltonian, k, d, alpha=1., optwith = optwith, solver = solver, maxiter = length(init_alpha:alpha_step:1)*maxiter, periodic = period, init_times = init_times)
        # results_qaoa["init_times"] = init_times

        # results_mcqaoa["init_times"] = [init_times]
        results_mcqaoa["result"] = nc_qaoa(hamiltonian, k, d, init_alpha=init_alpha, alpha_step=alpha_step, init_times=init_times)         

        # save(filename_mcqaoa, results_mcqaoa)

        dict_mcqaoa = Dict{String,Any}()
        if optwith == "optim"
            dict_mcqaoa["energies"] = Optim.minimum.(results_mcqaoa["result"])
            if solver == "cobyla"
                dict_mcqaoa["iterations"] = -ones(len(results_mcqaoa["result"]))
            else
                dict_mcqaoa["iterations"] = Optim.iterations.(results_mcqaoa["result"])
            end
        elseif optwith == "scipy"
            dict_mcqaoa["energies"] = [res[]["fun"] for res = results_mcqaoa["result"]]
            if solver == "cobyla"
                dict_mcqaoa["iterations"] = -ones(len(results_mcqaoa["result"]))
            else
                dict_mcqaoa["iterations"] = [res[]["nit"] for res = results_mcqaoa["result"]]
            end
        end

        # mark = Int8(round(init_alpha / 0.001))

        npzwrite("../compare_data/data/$(graph_type)/info_for_mc_qaoa_with_qubit$n-layer$k-initalpha-$init_alpha-inittyp-$init_type-alpha_step$alpha_step-exp$i-software$optwith-solver$solver-weight$weight-periodic$period-bounds$bound.npy", dict_mcqaoa)
        # npzwrite("../compare_data/data/$graph_type/info_for_qaoa_with_qubit$n-layer$k-initalpha-$init_alpha-inittyp-$init_type-alpha_step$alpha_step-exp$i-software$optwith-solver$solver-weight$weight-periodic$period-bounds$bound.npy", dict_qaoa)
        println("DONE\n")
        true
    end
    (parallel ? pmap : map)(i -> generate_experiment(i...), exp_tests_init)
end

rmprocs()