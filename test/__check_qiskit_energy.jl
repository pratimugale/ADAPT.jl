#= Run ADAPT on the Hubbard model with the qubit-excitation pool. =#

import ADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis

L = 2
u = 0.25

# BUILD OUT THE PROBLEM HAMILTONIAN: a periodic 1d Hubbard lattice at u=0.25
U = 4*u         # Dimensionless parameter u ≡ U/4|t|, and we'll set units so |t|=1.
H = ADAPT.Hamiltonians.hubbard_hamiltonian(L, U, -1.0, pbc=true)

# BUILD OUT THE QUBIT-EXCITATION POOL
pool, target_and_source = ADAPT.Pools.qubitexcitationpool(2L)
# pool = ADAPT.Pools.fullpauli(2L)

# CONSTRUCT A REFERENCE STATE
neel = "0110"^(L >> 1)
(L & 1 == 1) && (neel *= "01")
ψ0 = zeros(ComplexF64,2^(2L)); ψ0[parse(Int128, neel, base=2)+1] = 1.0

# INITIALIZE THE ANSATZ AND TRACE
ansatz = ADAPT.Ansatz(Float64, pool)
trace = ADAPT.Trace()

# SELECT THE PROTOCOLS
adapt = ADAPT.VANILLA
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores),
    ADAPT.Callbacks.ParameterTracer(),
    ADAPT.Callbacks.Printer(:energy, :selected_generator),
    ADAPT.Callbacks.ScoreStopper(1e-3),
    ADAPT.Callbacks.ParameterStopper(10),
]

# RUN THE ALGORITHM
ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
ψEND = ADAPT.evolve_state(ansatz, ψ0)

include("../qiskit_interface.jl")

energy_q = QiskitInterface.validate_energy(H, ansatz, ψ0, 2L)
println("Energy estimated in qiskit of ψEND = ", energy_q)

E0 = ADAPT.evaluate(H, ψEND)
println("Energy estimated in ADAPT of ψEND = ",E0)

ψEND_rev = QiskitInterface.revert_endianness(ψEND)
E0_r = ADAPT.evaluate(H, ψEND_rev)
println("Energy estimated in ADAPT of endianness-reverted ψEND = ",E0_r)
