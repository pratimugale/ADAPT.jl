import ..ADAPT
import PauliOperators: ScaledPauliVector

"""
    TETRISADAPT

Score pool operators by their initial gradients if they were to be appended to the pool.
TETRIS-ADAPT is a modified version of ADAPT-VQE in which multiple operators with disjoint 
support are added to the ansatz at each iteration. They are chosen by selecting from 
operators ordered in decreasing magnitude of gradients.
"""
struct TETRISADAPT{F} <: ADAPT.AdaptProtocol 
    gradient_threshold::F
end

ADAPT.typeof_score(::TETRISADAPT) = Float64

function support(spv::ScaledPauliVector)
    indices = Set{Int64}()
    for sp in spv
        op_indices=findall(x -> x != 'I',string(sp.pauli))
        union!(indices,op_indices)
    end
    return indices
end

function get_pauli_type(spv::ScaledPauliVector)
    # Get the Pauli string representation (e.g., "XX", "YY", "ZZ", "XY", etc.)
    # For ScaledPauliVector, we combine all Pauli terms
    pauli_strings = String[]
    for sp in spv
        pauli_str = string(sp.pauli)
        # Remove 'I' characters and get only the non-identity Paulis
        non_identity = filter(c -> c != 'I', pauli_str)
        if !isempty(non_identity)
            push!(pauli_strings, non_identity)
        end
    end
    return join(pauli_strings, " + ")  # If multiple terms, join them
end

function ADAPT.adapt!(
    ansatz::ADAPT.AbstractAnsatz,
    trace::ADAPT.Trace,
    adapt_type::TETRISADAPT,
    pool::ADAPT.GeneratorList,
    observable::ADAPT.Observable,
    reference::ADAPT.QuantumState,
    callbacks::ADAPT.CallbackList,
)
    println("TETRIS.adapt! called - Number of pool operators: $(length(pool))")
    # CALCULATE SCORES
    scores = ADAPT.calculate_scores(ansatz, adapt_type, pool, observable, reference)

    # CHECK FOR CONVERGENCE
    ε = eps(ADAPT.typeof_score(adapt_type))
    if all(score -> abs(score) < ε, scores)
        ADAPT.set_converged!(ansatz, true)
        return false
    end

    # MAKE SELECTION
    candidates = Dict(pool .=> scores)
    imap = Dict(pool .=> eachindex(pool))
    ops_to_add = Int64[]

    # remove candidate operators with scores below some threshold
    filter!(p-> p.second >= adapt_type.gradient_threshold, candidates)

    while !isempty(candidates)
        largest_score, G = findmax(candidates)
        G_support = support(G)
        
        # Print the selected operator's support and type
        G_pauli_type = get_pauli_type(G)
        println("Selected operator with score $(largest_score): type $(G_pauli_type), affects qubits $(sort(collect(G_support)))")
        
        # Check for candidates within 1% of the largest score
        # Note: p.first != G ensures we don't consider the selected operator again
        threshold_1percent = 0.99 * largest_score
        near_candidates = filter(p -> p.second >= threshold_1percent && p.first != G, candidates)
        
        if !isempty(near_candidates)
            println("Found $(length(near_candidates)) operator(s) within 1% of largest score ($(largest_score)):")
            for (op, score) in near_candidates
                op_support = support(op)
                op_pauli_type = get_pauli_type(op)
                overlap = intersect(G_support, op_support)
                overlap_str = isempty(overlap) ? " (disjoint)" : " (overlaps on qubits $(sort(collect(overlap))))"
                println("  - Operator with score $(score): type $(op_pauli_type), affects qubits $(sort(collect(op_support)))$overlap_str")
            end
        else
            println("No operators within 1% of largest score ($(largest_score))")
        end
        
        filter!(p-> isdisjoint(support(p.first), G_support), candidates)
        push!(ops_to_add, imap[G])
    end

    selected_indices = ops_to_add
    selected_scores = scores[selected_indices];
    selected_generators = pool[selected_indices];
    selected_parameters = zeros(ADAPT.typeof_parameter(ansatz), length(ops_to_add));
    
    # Calculate density and sum of gradients for selected operators
    if !isempty(selected_generators)
        # Get union of all qubits affected by selected operators
        all_affected_qubits = Set{Int64}()
        for gen in selected_generators
            union!(all_affected_qubits, support(gen))
        end
        
        # Get total number of qubits from any operator in the pool
        # ScaledPauliVector{N} where N is the number of qubits
        # We can get it from the string representation of a Pauli operator
        total_qubits = length(string(pool[1][1].pauli))  # Pauli string length = number of qubits
        
        # Calculate density
        density = length(all_affected_qubits) / total_qubits
        
        # Sum of gradients
        sum_gradients = sum(selected_scores)
        
        println("Selected operators analysis:")
        println("  - Number of operators selected: $(length(selected_generators))")
        println("  - Total qubits affected: $(length(all_affected_qubits)) out of $(total_qubits)")
        println("  - Density: $(round(density, digits=4)) ($(round(100*density, digits=2))%)")
        println("  - Sum of gradients: $(sum_gradients)")
        println("  - Affected qubits: $(sort(collect(all_affected_qubits)))")
    end

    # DEFER TO CALLBACKS
    data = ADAPT.Data(
        :scores => scores,
        :selected_index => selected_indices,
        :selected_score => selected_scores,
        :selected_generator => selected_generators,
        :selected_parameter => selected_parameters,
    )

    stop = false
    for callback in callbacks
        stop = stop || callback(data, ansatz, trace, adapt_type, pool, observable, reference)
        # Note that, as soon as `stop` is true, subsequent callbacks are short-circuited.
    end
    (stop || ADAPT.is_converged(ansatz)) && return false

    # PERFORM ADAPTATION
    for i in range(1,length(selected_generators))
        push!(ansatz, selected_generators[i] => selected_parameters[i])
    end
    ADAPT.set_optimized!(ansatz, false)
    return true
end

function ADAPT.calculate_score(
    ansatz::ADAPT.AbstractAnsatz,
    adapt_type::TETRISADAPT,
    generator::ADAPT.Generator,
    observable::ADAPT.Observable,
    reference::ADAPT.QuantumState,
)
    L = length(ansatz)
    candidate = deepcopy(ansatz)
    push!(candidate, generator => zero(ADAPT.typeof_parameter(ansatz)))
    return abs(ADAPT.partial(L+1, candidate, observable, reference))
end