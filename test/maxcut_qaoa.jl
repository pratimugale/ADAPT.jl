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

# EXAMPLE OF ANOTHER ERDOS-RENYI
#ne = 7
#g = Graphs.erdos_renyi(n, ne)

# EXTRACT MAXCUT FROM GRAPH
e_list = ADAPT.Hamiltonians.get_unweighted_maxcut(g)

# BUILD OUT THE PROBLEM HAMILTONIAN
H = ADAPT.Hamiltonians.maxcut_hamiltonian(n, e_list)

# ANOTHER WAY TO BUILD OUT THE PROBLEM HAMILTONIAN
#d = 3 # degree of regular graph
#H = ADAPT.Hamiltonians.MaxCut.random_regular_max_cut_hamiltonian(n, d)
println("Observable data type: ",typeof(H))

# EXACT DIAGONALIZATION (for accuracy assessment)
#= NOTE: Comment this out to avoid building exponentially-sized matrices!
    (but then you'll have to edit or remove the FloorStopper callback. =#
module Exact
    import ..H
    using LinearAlgebra
    Hm = Matrix(H); E, U = eigen(Hm) # NOTE: Comment out after first run when debugging.
    ψ0 = U[:,1]
    E0 = real(E[1])
end
println("Exact ground-state energy: ",Exact.E0)

# BUILD OUT THE POOL
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n)

# ANOTHER POOL OPTION
#pool = ADAPT.Pools.two_local_pool(n)

println("Generator data type: ", typeof(pool[1]))
println("Note: in the current ADAPT-QAOA implementation, the observable and generators must have the same type.")

# CONSTRUCT A REFERENCE STATE
ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0)

# INITIALIZE THE ANSATZ AND TRACE
ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(0.1, H)
# the first argument (a hyperparameter) can in principle be set to values other than 0.1
trace = ADAPT.Trace()

# SELECT THE PROTOCOLS
adapt = ADAPT.VANILLA # Can be changed to `ADAPT.Degenerate_ADAPT.DegenerateADAPT(1e-8) `
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)
    #= NOTE: Add `iterations=10` to set max iterations per optimization loop. =#

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores),
    ADAPT.Callbacks.ParameterTracer(),
    ADAPT.Callbacks.Printer(:energy, :selected_index, :selected_score),
    ADAPT.Callbacks.ScoreStopper(1e-3),
    ADAPT.Callbacks.ParameterStopper(100),
    ADAPT.Callbacks.FloorStopper(0.5, Exact.E0),
    ADAPT.Callbacks.SlowStopper(1.0, 3),
]

# RUN THE ALGORITHM
success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
println(success ? "Success!" : "Failure - optimization didn't converge.")

# VALIDATE ENERGY IN QISKIT
ψEND = ADAPT.evolve_state(ansatz, ψ0)

include("../qiskit_interface.jl")

energy_q = QiskitInterface.validate_energy(H, ansatz, ψ0, n)
println("Energy estimated in qiskit of ψEND = ", energy_q[1])

E0 = ADAPT.evaluate(H, ψEND)
println("Energy estimated in ADAPT of ψEND = ",E0)

# ψEND_rev = QiskitInterface.revert_endianness(ψEND)
# E0_r = ADAPT.evaluate(H, ψEND_rev)
# println("Energy estimated in ADAPT of endianness-reverted ψEND = ",E0_r)