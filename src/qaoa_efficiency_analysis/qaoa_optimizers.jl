# using Pkg
# Pkg.activate(".")

using LinearAlgebra
using SparseArrays
using Optim
using Random
using Distributions
using NPZ
using LineSearches
using Statistics
using SciPy

include("qaoa_utils.jl")

import Optim: retract!, project_tangent!

struct Periodic <: Manifold
    periods::Vector{Float64}
    function Periodic(v::Vector{Float64})
        @assert all(v .> 0.0)
        new(v)
    end
end

periods(p::Periodic) = p.periods

function retract!(p::Periodic, x)
    per = periods(p)
    x .%= per
    neg_ind = findall(x -> x < 0.0, x)
    correctors_int = ceil.(Int, abs.(x[neg_ind] ./ per[neg_ind]))
    x[neg_ind] .+= per[neg_ind] .* correctors_int
    x .%= per
    x
end

project_tangent!(p::Periodic, g, x) = g

"""
    qaoa(
    hamiltonian::Vector{T},
    k::Int,
    d::Dict;
    upper::Float64 = 2 * Float64(pi),
    alpha::Float64 = 1.0,
    init_state::Vector{ComplexF64} = fill(
        one(ComplexF64) / sqrt(length(hamiltonian)),
        length(hamiltonian),
    ),
    optwith::String = "optim",
    solver::String = "lbfgs",
    maxiter::Int64 = 10000,
    periodic::Bool = false,
    bounds::Bool = false,
    init_times::Vector{Float64} = rand(2 * k) .* vcat(fill(upper, k), fill(pi, k)),
)

Simulates QAOA energy evaluation and returns optimization results. 
It takes as arguments an objective Hamiltonian (`hamiltonian`), number of layers (`k`),
and a sparse matrix dictionary for Hamiltonian dimension (`d`). 

The function has 9 keywords:
* `upper`, upper bound of initial parameters value
* `alpha`, smoothing parameter for total hamiltonian ``H(α) = (1-α)H_mixer - αH_obj``. 
* `init_state`, initial state vector.
* `optwith`, choice of optmization package (`optim` or `scipy`). 
* `solver`, choice of type of opmization algorithm.
* `maxiter`, maximal number of iterations. 
* `periodic`, periodic object to pass if optimization is definided with periodic conditions.
* `bounds`, bounds for optimization space. 
* `init_times`, initial optimization parameters.
    
"""
function qaoa(
    hamiltonian::Vector{T},
    k::Int,
    d::Dict;
    upper::Float64 = 2 * Float64(pi),
    alpha::Float64 = 1.0,
    init_state::Vector{ComplexF64} = fill(
        one(ComplexF64) / sqrt(length(hamiltonian)),
        length(hamiltonian)
    ),
    optwith::String = "optim",
    solver::String = "lbfgs",
    maxiter::Int64 = 10000,
    periodic::Bool = false,
    bounds::Bool = false,
    init_times::Vector{Float64} = rand(2 * k) .* vcat(fill(upper, k), fill(pi, k)),
) where {T<:Real}
    @assert upper > 0.0

    n = length(hamiltonian)

    tmp_data = Dict{String,Any}(
        "state" => zeros(ComplexF64, n),
        "mul_vec" => zeros(ComplexF64, n),
        "v" => zeros(ComplexF64, 2 * n),
        "tmp_vec" => zeros(ComplexF64, n),
        "tmp_vec2" => zeros(ComplexF64, n),
        "d" => d,
    )

    periods = vcat(fill(upper, k), fill(pi, k))

    if optwith == "optim"
        fg! = (F, G, x) -> _fg!(F, G, hamiltonian, x, tmp_data, init_state, alpha)
        opt = Optim.Options(
            g_tol = 1e-5,
            x_tol = 1e-5,
            f_tol = 1e-5,
            allow_f_increases = true,
            iterations = maxiter
        )
        if solver == "lbfgs"
            optimizer = LBFGS()
            if periodic == true
                optimizer = LBFGS(manifold = Periodic(periods))
            end
        elseif solver == "gd"
            optimizer = GradientDescent()
            if periodic == true
                optimizer = GradientDescent(manifold = Periodic(periods))
            end
        elseif solver == "nm"
            optimizer = NelderMead()
        end
        return Optim.optimize(Optim.only_fg!(fg!), init_times, optimizer, opt)

    elseif optwith == "scipy"
        function _scipy_fg(x)
            gradient = zeros(length(x))
            energy = _fg!(1, gradient, hamiltonian, x, tmp_data, init_state, alpha)
            return energy, gradient
        end

        if bounds == true
            b = SciPy.optimize.Bounds(0, Float64(pi), keep_feasible = false)
        else
            b = nothing
        end

        method = "L-BFGS-B"

        if solver == "lbfgs"
            method = "L-BFGS-B"
        elseif solver == "cobyla"
            method = "COBYLA"
        elseif solver == "nm"
            method = "Nelder-Mead"
        end
        opt = Dict("maxiter" => maxiter)
        return SciPy.optimize.minimize(
            _scipy_fg,
            init_times,
            method = method,
            jac = true,
            options = opt,
            bounds = b,
        )
    end
end


"""
    nc_qaoa(
    hamiltonian::Vector{T},
    kmax::Int,
    d::Dict;
    upper::Float64 = 2 * Float64(pi),
    init_alpha::Float64 = 0.1,
    alpha_step::Float64 = 0.01,
    optwith::String = "optim",
    solver::String = "lbfgs",
    maxiter::Int64 = 50,
    periodic::Bool = false,
    init_times::Vector{Float64} = rand(2 * k) .* vcat(fill(upper, k), fill(pi, k)),
)

Simulates NC-QAOA energy evaluation and returns optimization results. 
It runs QAOA varying the alpha value trought several 
It takes as arguments an objective Hamiltonian (`hamiltonian`), maximal number of layers (`k`),
and a sparse matrix dictionary for Hamiltonian dimension (`d`). 

The function has 8 keywords:
* `upper`, upper bound of initial parameters value
* `init_alpha`, initial inital alpha value.
* `alpha_step`, alpha increment.
* `optwith`, choice of optmization package (`optim` or `scipy`). 
* `solver`, choice of type of opmization algorithm.
* `maxiter`, maximal number of iterations. 
* `periodic`, periodic object to pass if optimization is definided with periodic conditions.
* `init_times`, initial optimization parameters.

"""
function nc_qaoa(
    hamiltonian::Vector{T},
    kmax::Int,
    d::Dict;
    upper::Float64 = 2 * Float64(pi),
    init_alpha::Float64 = 0.1,
    alpha_step::Float64 = 0.01,
    optwith::String = "optim",
    solver::String = "lbfgs",
    maxiter::Int64 = 50,
    periodic::Bool = false,
    init_times::Vector{Float64} = rand(2 * k) .* vcat(fill(upper, k), fill(pi, k)),
) where {T<:Real}

    @assert upper > 0.0

    results = []
    for alpha = init_alpha:alpha_step:1
        res = qaoa(
            hamiltonian,
            kmax,
            d,
            alpha = alpha,
            optwith = optwith,
            solver = solver,
            maxiter = maxiter,
            periodic = periodic,
            init_times = init_times,
        )
        push!(results, res)
        init_times = Optim.minimizer(res)
    end
    return results
end

"""
    qaoa_trajectories_periodic(
    hamiltonian::Vector{T},
    kmax::Int;
    upper::Float64 = 2 * pi,
    kmin::Int = 2,
    d::Dict = Dict(),
    level_repeat::Int = 1,
    alpha::Float64 = 1.0,
    max_samples::Int = 1,
    method::String = "zero_rand",
    solver::String = "lbfgs",
    init_angles::Vector{Float64} = rand(2 * kmin) .*
                                   vcat(fill(upper, kmin), fill(pi, kmin)),
    init_state::Vector{ComplexF64} = fill(
        one(ComplexF64) / sqrt(length(hamiltonian)),
        length(hamiltonian),
    ),
) 

Simulates T-QAOA energy evaluation and returns optimization results. 
It uses the previous optimized angles as initial paramenters for the next layer.
It takes as arguments an objective Hamiltonian (`hamiltonian`) and maximal number of layers (`k`).

The function has 8 keywords:
* `upper`, upper bound of initial parameters value
* `kmin`, minimal number of layers for initialiation.
* `d`, a dictionary.
* `level_repeat`, number of times to repeat evaluation for a given layer number.
* `alpha`, alpha value.
* `max_samples`, number of samples for initial parameters.
* `method`, choice of initial paramenters sampling method.
* `solver`, choice of type of opmization algorithm.
* `init_times`, initial optimization parameters.
* `init_state`, initial vector state.

"""
function qaoa_trajectories_periodic(
    hamiltonian::Vector{T},
    kmax::Int;
    upper::Float64 = 2 * pi,
    kmin::Int = 2,
    d::Dict = Dict(),
    level_repeat::Int = 1,
    alpha::Float64 = 1.0,
    max_samples::Int = 1,
    method::String = "zero_rand",
    solver::String = "lbfgs",
    init_angles::Vector{Float64} = rand(2 * kmin) .*
                                   vcat(fill(upper, kmin), fill(pi, kmin)),
    init_state::Vector{ComplexF64} = fill(
        one(ComplexF64) / sqrt(length(hamiltonian)),
        length(hamiltonian),
    ),
) where {T<:Real}

    @assert kmax >= kmin >= 1
    @assert upper > 0.0
    @assert length(init_angles) == 2 * kmin

    n = length(hamiltonian)
    tmp_data = Dict{String,Any}(
        "state" => zeros(ComplexF64, n),
        "mul_vec" => zeros(ComplexF64, n),
        "v" => zeros(ComplexF64, 2 * n),
        "tmp_vec" => zeros(ComplexF64, n),
        "tmp_vec2" => zeros(ComplexF64, n),
        "d" => d,
    )


    fg! = (F, G, x) -> _fg!(F, G, hamiltonian, x, tmp_data, init_state, alpha)

    fg_optim! = Optim.only_fg!(fg!)

    opt = Optim.Options(g_tol = 1e-5, x_tol = 1e-5, f_tol = 1e-5, allow_f_increases = true)

    converged = false
    results = Any[]
    while !converged
        results = Any[]
        best_t = Float64[] 
        converged = true
        for k = kmin:kmax
            periods = vcat(fill(upper, k), fill(pi, k))
            if solver == "lbfgs"
                # optimizer = LBFGS(manifold=Periodic(periods))
                optimizer = LBFGS()
            elseif solver == "nm"
                optimizer = NelderMead()
            elseif solver == "gd"
                optimizer = GradientDescent(manifold = Periodic(periods))
            else
                println("solver not found: $solver")
            end
            # optimizer = LBFGS(manifold=Periodic(periods), alphaguess=InitialStatic(alpha=0.00001))            
            res_tmp = []
            init_times_vec = Array{Float64}[]
            if k == kmin
                init_times_vec = [init_angles]
            elseif method == "random"
                prev_k = k - 1
                samples = [
                    vcat(
                        best_t[1:prev_k],
                        rand() * upper,
                        best_t[prev_k+1:end],
                        rand() * pi,
                    ) for _ = 1:max_samples
                ] #HACK
                f_values = (t -> fg!(1, nothing, t)).(samples)
                init_times_vec =
                    samples[sortperm(f_values)[1:minimum([max_samples, level_repeat])]]
            elseif method == "zero_rand"
                prev_k = k - 1
                samples = [
                    vcat(best_t[1:prev_k], rand() * upper, best_t[prev_k+1:end], 0) for
                    _ = 1:max_samples
                ] #HACK
                f_values = (t -> fg!(1, nothing, t)).(samples)
                init_times_vec =
                    samples[sortperm(f_values)[1:minimum([max_samples, level_repeat])]] #minimum([max_samples,level_repeat])]
            elseif method == "interp"
                prev_k = k - 1
                old_gammas = prepend!(push!(copy(best_t[1:prev_k]), 0), 0)
                old_betas = prepend!(push!(copy(best_t[prev_k+1:end]), 0), 0)
                new_gammas = [
                    (i) / (prev_k) * old_gammas[i] +
                    ((prev_k - i + 2) / (prev_k)) * old_gammas[i+1] for i = 1:k
                ]
                new_betas = [
                    (i) / (prev_k) * old_betas[i] +
                    ((prev_k - i + 2) / (prev_k)) * old_betas[i+1] for i = 1:k
                ]
                init_times_vec = [vcat(new_gammas, new_betas)]
            elseif method == "interp_zero"
                prev_k = k - 1
                old_gammas = prepend!(push!(copy(best_t[1:prev_k]), 0), 0)
                new_gammas = [
                    (i) / (prev_k) * old_gammas[i] +
                    ((prev_k - i + 2) / (prev_k)) * old_gammas[i+1] for i = 1:k
                ]
                init_times_vec = [vcat(new_gammas, best_t[prev_k+1:end], 0)]
            end

            res_tmp = (it -> Optim.optimize(fg_optim!, it, optimizer, opt)).(init_times_vec)
            res_tmp_filtered = filter(Optim.converged, res_tmp)
            if length(res_tmp_filtered) == 0
                converged = false
                println("Failed qaoa_trajectories_periodic $k $level_repeat")
                break
            end
            _, pos = findmin(Optim.minimum.(res_tmp_filtered))
            # if log10(maximum(abs.(Optim.minimizer(res_tmp_filtered[pos])))) > 6
            #     @warn "$k Very high input: $(log10(maximum(abs.(Optim.minimizer(res_tmp_filtered[pos])))))"
            # end
            push!(results, res_tmp_filtered[pos])
            best_t = Optim.minimizer(res_tmp_filtered[pos])
            # best_t = ((best_t .% periods) .+ periods) .% periods
        end
    end
    results
end
