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

"""
    is_mixer_operator(op::ScaledPauliVector, n::Int)

Check if an operator is the QAOA mixer (all X on all qubits).
The mixer has length n and each element is X on a different qubit.
"""
function is_mixer_operator(op::ScaledPauliVector, n::Int)
    if length(op) != n
        return false
    end
    # Check that each element is X on exactly one qubit
    for sp in op
        pauli_str = string(sp.pauli)
        # Count non-I characters (should be exactly 1 for single X)
        non_identity_count = count(c -> c != 'I', pauli_str)
        if non_identity_count != 1
            return false
        end
        # Check that it's X (not Y or Z)
        x_positions = findall(c -> c == 'X' || c == 'x', pauli_str)
        if length(x_positions) != 1
            return false
        end
    end
    return true
end

"""
    is_single_x_operator(op::ScaledPauliVector)

Check if an operator is a single-qubit X operator.
Single X has length 1 and the pauli has X on exactly one qubit.
"""
function is_single_x_operator(op::ScaledPauliVector)
    if length(op) != 1
        return false
    end
    sp = op[1]
    pauli_str = string(sp.pauli)
    # Count non-I characters (should be exactly 1)
    non_identity_count = count(c -> c != 'I', pauli_str)
    if non_identity_count != 1
        return false
    end
    # Check that it's X (not Y or Z)
    x_positions = findall(c -> c == 'X' || c == 'x', pauli_str)
    return length(x_positions) == 1
end

"""
    find_alternative_operators(G, G_support, near_candidates)

Find alternative operators that follow the pattern:
- If G affects [a, b], look for operators affecting [a, c] and [b, d] that are disjoint.
- Returns (found, alternative_ops) where found is Bool and alternative_ops is a vector of operators.
"""
function find_alternative_operators(G, G_support, near_candidates)
    G_support_list = sort(collect(G_support))
    
    # Only apply this logic for two-qubit operators
    if length(G_support_list) != 2
        return (false, [])
    end
    
    a, b = G_support_list[1], G_support_list[2]
    
    # Find operators that overlap with exactly one qubit from G
    candidates_overlapping_a = []
    candidates_overlapping_b = []
    
    for (op, score) in near_candidates
        op_support = support(op)
        overlap = intersect(G_support, op_support)
        
        # Must overlap with exactly one qubit from G
        if length(overlap) == 1
            if a in overlap
                push!(candidates_overlapping_a, (op, score))
            elseif b in overlap
                push!(candidates_overlapping_b, (op, score))
            end
        end
    end
    
    # Try to find a pair: one overlapping with a, one overlapping with b, and they are disjoint
    for (op1, score1) in candidates_overlapping_a
        op1_support = support(op1)
        for (op2, score2) in candidates_overlapping_b
            op2_support = support(op2)
            # Check if op1 and op2 are disjoint
            if isdisjoint(op1_support, op2_support)
                println("Found alternative pattern: replacing operator on [$a, $b] with operators on $(sort(collect(op1_support))) and $(sort(collect(op2_support)))")
                return (true, [op1, op2])
            end
        end
    end
    
    return (false, [])
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

    # Get number of qubits from the pool
    n = length(string(pool[1][1].pauli))

    while !isempty(candidates)
        largest_score, G = findmax(candidates)
        G_support = support(G)
        
        # Print the selected operator directly and its support
        println("Selected operator with score $(largest_score): $G")
        println("  Affects qubits: $(sort(collect(G_support)))")
        
        # Check for candidates within 1% of the largest score
        # Note: p.first != G ensures we don't consider the selected operator again
        threshold_1percent = 0.99 * largest_score
        near_candidates = filter(p -> p.second >= threshold_1percent && p.first != G, candidates)
        
        if !isempty(near_candidates)
            println("Found $(length(near_candidates)) operator(s) within 1% of largest score ($(largest_score)):")
            for (op, score) in near_candidates
                op_support = support(op)
                overlap = intersect(G_support, op_support)
                overlap_str = isempty(overlap) ? " (disjoint)" : " (overlaps on qubits $(sort(collect(overlap))))"
                println("  - Operator with score $(score): $op, affects qubits $(sort(collect(op_support)))$overlap_str")
            end
        else
            println("No operators within 1% of largest score ($(largest_score))")
        end
        
        # Check if G is mixer or single X - if yes, use G as normal
        # Otherwise, check for alternative operators
        operators_to_add_this_iteration = [G]
        
        if !is_mixer_operator(G, n) && !is_single_x_operator(G)
            # Try to find alternative operators
            found_alternatives, alternative_ops = find_alternative_operators(G, G_support, near_candidates)
            
            if found_alternatives
                # Use alternative operators instead of G
                operators_to_add_this_iteration = alternative_ops
                println("Using alternative operators instead of the original selection")
                # Remove G from candidates since we're not using it
                delete!(candidates, G)
            end
        else
            println("Operator is mixer or single X - using standard selection")
        end
        
        # Remove all selected operators and overlapping operators from candidates
        for op in operators_to_add_this_iteration
            op_support = support(op)
            # Remove the operator itself from candidates
            delete!(candidates, op)
            # Remove all operators that overlap with this operator
            filter!(p-> isdisjoint(support(p.first), op_support), candidates)
            push!(ops_to_add, imap[op])
        end
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