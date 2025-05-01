#= Run ADAPT-QAOA on a MaxCut Hamiltonian. =#

import Graphs
import ADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis
import LinearAlgebra: norm

import Random; Random.seed!(0)

# DEFINE A GRAPH
n = 6

# EXAMPLE OF ERDOS-RENYI GRAPH
prob = 0.5
g = Graphs.erdos_renyi(n, prob)

# EXTRACT MAXCUT FROM GRAPH
e_list = ADAPT.Hamiltonians.get_unweighted_maxcut(g)

# BUILD OUT THE PROBLEM HAMILTONIAN
H = ADAPT.Hamiltonians.maxcut_hamiltonian(n, e_list)

println("Observable data type: ",typeof(H))

# BUILD OUT THE POOL
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n)

# ANOTHER POOL OPTION
# pool = ADAPT.Pools.two_local_pool(n)

println("Generator data type: ", typeof(pool[1]))
println("Note: in the current ADAPT-QAOA implementation, the observable and generators must have the same type.")

# CONSTRUCT A REFERENCE STATE
ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0)

# INITIALIZE THE ANSATZ AND TRACE
gamma0 = 0.1; println("gamma0 = $(gamma0)")
ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(gamma0, H)
# the first argument (a hyperparameter) can in principle be set to values other than 0.1
trace = ADAPT.Trace()

# SELECT THE PROTOCOLS
adapt = ADAPT.TETRIS_ADAPT.TETRISADAPT(1e-3) #ADAPT.Degenerate_ADAPT.DegenerateADAPT(1e-3) #ADAPT.TETRIS_ADAPT.TETRISADAPT(1e-3) 
# ADAPT.VANILLA # Can be changed to `ADAPT.Degenerate_ADAPT.DegenerateADAPT(1e-8) `
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)
    #= NOTE: Add `iterations=10` to set max iterations per optimization loop. =#

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores),
    ADAPT.Callbacks.ParameterTracer(),
    ADAPT.Callbacks.Printer(:energy,:selected_generator),
    ADAPT.Callbacks.ScoreStopper(1e-3),
    ADAPT.Callbacks.ParameterStopper(100),
    # ADAPT.Callbacks.FloorStopper(0.5, Exact.E0),
    ADAPT.Callbacks.SlowStopper(1.0, 3),
]

# RUN THE ALGORITHM
success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
println(success ? "Success!" : "Failure - optimization didn't converge.")

println(ansatz)
