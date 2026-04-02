import ..ADAPT
import PauliOperators: ScaledPauliVector

"""
    TETRISADAPT

Score pool operators by their initial gradients if they were to be appended to the pool.
TETRIS-ADAPT is a modified version of ADAPT-VQE in which multiple operators with disjoint 
support are added to the ansatz at each iteration. They are chosen by selecting from 
operators ordered in decreasing magnitude of gradients.

If use_kamis is true, uses KaMIS mmwis algorithm for maximum weight independent set selection.
Otherwise, uses the original greedy selection method.
"""
struct TETRISADAPT{F} <: ADAPT.AdaptProtocol
    gradient_threshold::F
    use_kamis::Bool
    kamis_path::String
    kamis_seed::Int
    percent_tail_ends_removed::Float64
end

# Constructor with default values
TETRISADAPT(gradient_threshold::F; use_kamis::Bool=false, kamis_path::String="", kamis_seed::Int=0, percent_tail_ends_removed::Float64=0.0) where F =
    TETRISADAPT{F}(gradient_threshold, use_kamis, kamis_path, kamis_seed, percent_tail_ends_removed)

ADAPT.typeof_score(::TETRISADAPT) = Float64

"""
    AdaptationStopper(max_adaptations::Int)

Converge once the number of adaptations reaches a maximum.

Called for `adapt!` only. Requires a preceding `Tracer`.

# Parameters
- `max_adaptations`: the maximum number of adaptations allowed
"""
struct AdaptationStopper <: ADAPT.AbstractCallback
    max_adaptations::Int
end

function (stopper::AdaptationStopper)(
    ::ADAPT.Data, ansatz::ADAPT.AbstractAnsatz, trace::ADAPT.Trace,
    ::ADAPT.AdaptProtocol, ::ADAPT.GeneratorList, ::ADAPT.Observable, ::ADAPT.QuantumState,
)
    haskey(trace, :adaptation) || return false
    if length(trace[:adaptation]) >= stopper.max_adaptations
        ADAPT.set_converged!(ansatz, true)
        println("Algorithm terminated: Number of adaptations ($(length(trace[:adaptation]))) reached maximum ($(stopper.max_adaptations))")
    end
    return false
end

function support(spv::ScaledPauliVector)
    indices = Set{Int64}()
    for sp in spv
        op_indices = findall(x -> x != 'I', string(sp.pauli))
        union!(indices, op_indices)
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
    remove_tail_ends(selected_indices, selected_generators, selected_scores, percent_tail_ends_removed)

Remove operators from the tail (smallest gradients) such that their cumulative sum is at most
percent_tail_ends_removed% of the total gradient sum. This ensures we never remove more than x%.
If all operators would be removed, keeps the one with highest gradient.
Returns (filtered_indices, filtered_generators, filtered_scores, num_removed).
"""
function remove_tail_ends(selected_indices, selected_generators, selected_scores, percent_tail_ends_removed)
    if percent_tail_ends_removed <= 0.0 || isempty(selected_scores)
        return selected_indices, selected_generators, selected_scores, 0
    end

    total_sum = sum(selected_scores)
    limit = (percent_tail_ends_removed / 100.0) * total_sum

    # Sort indices by score ascending (smallest first)
    sorted_order = sortperm(selected_scores)

    # Accumulate from smallest, mark for removal while cumulative <= limit
    cumulative = 0.0
    remove_set = Set{Int}()
    for idx in sorted_order
        score = selected_scores[idx]
        if cumulative + score <= limit
            cumulative += score
            push!(remove_set, idx)
        else
            break
        end
    end

    # If all would be removed, keep the one with highest gradient
    if length(remove_set) >= length(selected_scores)
        max_idx = argmax(selected_scores)
        println("  Tail-end removal: all operators would be removed, keeping highest gradient operator")
        return [selected_indices[max_idx]], [selected_generators[max_idx]], [selected_scores[max_idx]], length(selected_scores) - 1
    end

    # Build keep mask
    keep_mask = [!(i in remove_set) for i in 1:length(selected_scores)]
    num_removed = length(remove_set)

    filtered_indices = selected_indices[keep_mask]
    filtered_generators = selected_generators[keep_mask]
    filtered_scores = selected_scores[keep_mask]

    println("  Tail-end removal: removed $num_removed operators (cumulative=$(round(cumulative, digits=6)), limit=$(round(limit, digits=6)), $(percent_tail_ends_removed)% of sum=$total_sum)")

    return filtered_indices, filtered_generators, filtered_scores, num_removed
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
        println("Algorithm terminated: All operator scores below machine epsilon (converged)")
        return false
    end

    # MAKE SELECTION
    num_operators_removed = 0
    if adapt_type.use_kamis
        # Use KaMIS mmwis for maximum weight independent set selection
        println("Using KaMIS mmwis for operator selection")
        selected_indices, selected_generators, selected_scores =
            select_operators_with_kamis(pool, scores, adapt_type.gradient_threshold;
                kamis_path=adapt_type.kamis_path,
                seed=adapt_type.kamis_seed)
        # Remove tail-end operators if configured
        selected_indices, selected_generators, selected_scores, num_operators_removed =
            remove_tail_ends(selected_indices, selected_generators, selected_scores, adapt_type.percent_tail_ends_removed)
        selected_parameters = zeros(ADAPT.typeof_parameter(ansatz), length(selected_indices))
    else
        # Original greedy selection method
        candidates = Dict(pool .=> scores)
        imap = Dict(pool .=> eachindex(pool))
        ops_to_add = Int64[]

        # remove candidate operators with scores below some threshold
        filter!(p -> p.second >= adapt_type.gradient_threshold, candidates)

        while !isempty(candidates)
            largest_score, G = findmax(candidates)
            G_support = support(G)
            filter!(p -> isdisjoint(support(p.first), G_support), candidates)
            push!(ops_to_add, imap[G])
        end

        selected_indices = ops_to_add
        selected_scores = scores[selected_indices]
        selected_generators = pool[selected_indices]
        selected_parameters = zeros(ADAPT.typeof_parameter(ansatz), length(ops_to_add))
    end

    # Calculate density and sum of gradients for selected operators
    sum_gradients = 0.0
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

    # Add sum of gradients if available
    if sum_gradients > 0.0
        data[:sum_gradients] = sum_gradients
    end

    # Track number of operators removed by tail-end filtering
    data[:num_operators_removed] = num_operators_removed

    stop = false
    for callback in callbacks
        stop = stop || callback(data, ansatz, trace, adapt_type, pool, observable, reference)
        # Note that, as soon as `stop` is true, subsequent callbacks are short-circuited.
    end
    if stop || ADAPT.is_converged(ansatz)
        if stop
            println("Algorithm terminated: Callback requested termination")
        else
            println("Algorithm terminated: Ansatz marked as converged")
        end
        return false
    end

    # PERFORM ADAPTATION
    for i in range(1, length(selected_generators))
        push!(ansatz, selected_generators[i] => selected_parameters[i])
    end
    ADAPT.set_optimized!(ansatz, false) # because we want to optimize all the parameters again after adding the new operators 
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
    return abs(ADAPT.partial(L + 1, candidate, observable, reference))
end