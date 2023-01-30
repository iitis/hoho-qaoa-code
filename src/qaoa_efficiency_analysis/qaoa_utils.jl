# using Pkg
# Pkg.activate(".")

using LinearAlgebra
using SparseArrays
using Optim
using Random
using NPZ
include("sparse_generator_loader.jl")

##
# n is number of edges ,

function _state!(
    hamiltonian::Vector{Float64},
    times::Vector{Float64},
    tmp_data::Dict,
    init_state::Vector{ComplexF64},
)
    k = div(length(times), 2)
    qubits_no = Int(log2(length(hamiltonian)))

    state = tmp_data["state"]
    mulvec = tmp_data["mul_vec"]

    state .= init_state

    for (p, r) in zip(times[1:k], times[(k+1):end])
        @inbounds broadcast!((x, y) -> x * exp(-1im * p * y), state, state, hamiltonian)
        for q = 1:qubits_no
            @inbounds mul!(
                mulvec,
                sparse_up_1qubit!(tmp_data["v"], qubits_no, q, r, tmp_data["d"]),
                state,
            )
            mulvec, state = state, mulvec
        end
    end
    state
end

function _energy!(
    times::Vector{Float64},
    tmp_data::Dict,
    init_state::Vector{ComplexF64},
    hamiltonian::Vector{Float64},
    ham_energy::Vector{Float64} = nothing,
)
    state = _state!(hamiltonian, times, tmp_data, init_state)
    if ham_energy == nothing
        @inbounds broadcast!((x, y) -> abs2(x) * y, state, state, hamiltonian)
    else
        @inbounds broadcast!((x, y) -> abs2(x) * y, state, state, ham_energy)
    end
    abs(sum(state))
end

function _fg!(
    F,
    gradient,
    hamiltonian::Vector{Float64},
    times,
    tmp_data::Dict,
    init_state::Vector{ComplexF64},
    alpha::Float64 = 0.0,
)
    k = div(length(times), 2)
    qubits_no = Int(log2(length(hamiltonian)))
    tmp_vec = tmp_data["tmp_vec"]
    tmp_vec2 = tmp_data["tmp_vec2"]
    d = tmp_data["d"]
    v = tmp_data["v"]

    state = _state!(hamiltonian, times, tmp_data, init_state)
    # Check redudancy
    mulvec = state === tmp_data["mul_vec"] ? tmp_data["state"] : tmp_data["mul_vec"]


    copy!(tmp_vec, state)
    @inbounds broadcast!((x, y) -> alpha * x * y, state, hamiltonian, state)

    for q = 1:qubits_no
        @inbounds mul!(mulvec, sparse_up_1qubit_x!(v, qubits_no, q, d), tmp_vec)
        @inbounds broadcast!((x, y) -> -(1 - alpha) * x + y, state, mulvec, state)
    end


    if F != nothing
        @inbounds broadcast!((x, y) -> conj(x) * y, mulvec, state, tmp_vec)
        F = real(sum(mulvec))
    end
    if gradient != nothing
        for (i, p, r) in zip(k:-1:1, times[k:-1:1], times[2*k:-1:(k+1)])
            tmp_vec2 .= 0.0
            for q = 1:qubits_no

                @inbounds mul!(
                    mulvec,
                    sparse_up_1qubit_special!(v, qubits_no, q, d),
                    tmp_vec,
                )
                # one can use dot: the differnce is neglible
                @inbounds broadcast!(
                    (x, y, z) -> x + conj(y) * z,
                    tmp_vec2,
                    tmp_vec2,
                    mulvec,
                    state,
                )
                @inbounds mul!(
                    tmp_vec,
                    sparse_up_1qubit_der_dag!(v, qubits_no, q, r, d),
                    mulvec,
                )
                @inbounds mul!(mulvec, sparse_up_1qubit_dag!(v, qubits_no, q, r, d), state)
                mulvec, state = state, mulvec
            end
            gradient[k+i] = 2 * real(sum(tmp_vec2))

            tmp_vec2 .= 0.0
            @inbounds broadcast!(
                (x, y) -> exp(1im * p * y) * x,
                tmp_vec,
                tmp_vec,
                hamiltonian,
            )
            @inbounds broadcast!((x, y) -> exp(1im * p * y) * x, state, state, hamiltonian)
            @inbounds broadcast!(
                (x, y, z) -> 1im * conj(x) * y * z,
                tmp_vec2,
                tmp_vec,
                hamiltonian,
                state,
            )
            gradient[i] = 2 * real(sum(tmp_vec2))
        end
    end
    F
end
