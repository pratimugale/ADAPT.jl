import ADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, Pauli
import CSV
import DataFrames
import Serialization
import LinearAlgebra: norm

n = 5; gamma0 = 0.1

# READ IN HAMILTONIAN
serialized_H = "adaptqaoa_Hamiltonian_n_"*string(n)*"_gamma0_"*string(gamma0)
H_asdict = Serialization.deserialize(serialized_H)
H_spv = ScaledPauliVector{n}()
for (pauli,coeff) in H_asdict
    term = coeff*Pauli(pauli)
    push!(H_spv,term)
end
# Wrap in a QAOAObservable view.
H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv)

# READ IN ADAPT-QAOA RESULTS
results_file = "adaptqaoa_results_n_"*string(n)*"_gamma0_"*string(gamma0)*".csv"
csv = CSV.File(results_file); my_df = DataFrames.DataFrame(csv)

# BUILD OUT THE OPERATOR POOL
# pooltype = my_df[!,:pooltype][1]; println(pooltype) 
# if pooltype=="qaoa_double_pool" 
    
# end
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n)

# INITIALIZE THE ANSATZ 
ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0) # initialize ψ0
ansatz = ADAPT.ADAPT_QAOA.TetrisQAOAAnsatz(gamma0, pool, H) # initialize ansatz

# RECONSTRUCT ANSATZ
for row in eachrow(my_df)
    push!(ansatz.γ_values, row.:gamma_coeff)
    push!(ansatz.γ_layers, 1+length(ansatz.parameters))

    @assert row.:selected_index[1] == '[' && row.:selected_index[end] == ']'
    indices = [parse(Int, d) for d in split(strip(row.:selected_index, ['[', ']']), ",")]
    for index in indices
        push!(ansatz.generators, pool[index])
    end

    @assert row.:beta_coeff[1] == '[' && row.:beta_coeff[end] == ']'
    beta_coefficients = [parse(Float64, d) for d in split(strip(row.:beta_coeff, ['[', ']']), ",")]
    append!(ansatz.parameters, beta_coefficients)
end #= <- this is your reconstructed ansatz =#

# TEST: EVALUATE FINAL ENERGY - SHOULD MATCH LAST ENERGY
ψEND = ADAPT.evolve_state(ansatz, ψ0)
E_final = ADAPT.evaluate(H, ψEND)
println("final energy = $E_final")