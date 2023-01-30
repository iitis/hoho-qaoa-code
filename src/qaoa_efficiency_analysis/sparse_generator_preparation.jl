using Pkg
Pkg.activate(".")

include("sparse_generator_loader.jl")

function generator(n::Int)
    @assert 32 >= n >= 1
    for k = 1:n
        m = _1qubit_gen(k, n)
        if k == 1
            npzwrite("sparse_1qubitgate_data/colptr_$n.npz", m.colptr)
        end
        npzwrite("sparse_1qubitgate_data/rowval_$n-$k.npz", m.rowval)
        npzwrite(
            "sparse_1qubitgate_data/topright_$n-$k.npz",
            Vector{Int32}(findall(x -> x == -1im, m.nzval)),
        )
    end
    nothing
end

n = parse(Int, ARGS[1]) # number of nodes
generator(n)
