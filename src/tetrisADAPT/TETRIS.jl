import ..ADAPT
import PauliOperators: ScaledPauliVector

"""
    TETRISADAPT

Score pool operators by their initial gradients if they were to be appended to the pool.
TETRIS-ADAPT is a modified version of ADAPT-VQE in which multiple operators with disjoint 
support are added to the ansatz at each iteration. They are chosen by selecting from 
operators ordered in decreasing magnitude of gradients if use_kamis is false. If use_kamis 
is true, then the incompatibility problem is solved using KaMIS mmwis (Memetic Maximum 
Weight Independent Set Algorithm) and the operators are selected based on the solution.
"""
struct TETRISADAPT{F} <: ADAPT.AdaptProtocol 
    gradient_threshold::F
    use_kamis::Bool
    kamis_seed::Int
    percent_tail_ends_removed::Float64
end

# Constructor
function TETRISADAPT(
    gradient_threshold::F; # we can still run without Kamis to provide backward compatibility
    use_kamis::Bool = false,
    kamis_seed::Int = 42,
    percent_tail_ends_removed::Float64 = 0.0
) where F
    if percent_tail_ends_removed < 0 || percent_tail_ends_removed > 1
        throw(ArgumentError("percent_tail_ends_removed must be between 0 and 1 (inclusive)"))
    end

    # check if the mmwis binary exists
    if use_kamis && !isfile("external/KaMIS/deploy/mmwis")
        @warn("KaMIS mmwis binary not found at external/KaMIS/deploy/mmwis. \nMake sure you have built KaMIS and the binary is in the correct location. \nSee https://github.com/KaMIS/KaMIS for instructions.")
    end
    
    if use_kamis && kamis_seed < 0
        throw(ArgumentError("kamis_seed must be a non-negative integer"))
    end

    return TETRISADAPT{F}(gradient_threshold, use_kamis, kamis_seed, percent_tail_ends_removed)
end

ADAPT.typeof_score(::TETRISADAPT) = Float64

function ADAPT.adapt!(
    ansatz::ADAPT.AbstractAnsatz,
    trace::ADAPT.Trace,
    adapt_type::TETRISADAPT,
    pool::ADAPT.GeneratorList,
    observable::ADAPT.Observable,
    reference::ADAPT.QuantumState,
    callbacks::ADAPT.CallbackList,
)
    # CALCULATE SCORES
    scores = ADAPT.calculate_scores(ansatz, adapt_type, pool, observable, reference)
    
    # Note that calculate_score (defined below) already returns the absolute value of the partial derivative.
    # We do not need to perform abs() again to filter out operators based on magnitude.

    # CHECK FOR CONVERGENCE
    ε = eps(ADAPT.typeof_score(adapt_type))
    if all(score -> abs(score) < ε, scores)
        ADAPT.set_converged!(ansatz, true)
        return false
    end

    # Perform selection based on chosen method
    if adapt_type.use_kamis
        # Perform selection using KaMIS
        @info "Performing selection using KaMIS"
        # TODO: it seems that we will filter again in the next function call. Modify this layer
        selected_indices, selected_generators, selected_scores = select_operators_with_kamis(
            pool, scores, adapt_type.gradient_threshold,
            seed=adapt_type.kamis_seed)
        
        # Remove tail ends, if configured
        if adapt_type.percent_tail_ends_removed > 0
            selected_indices, selected_generators, 
            selected_scores, num_operators_removed = remove_tail_ends(
                selected_indices, selected_generators, selected_scores, 
                adapt_type.percent_tail_ends_removed)
        end

        selected_parameters = zeros(ADAPT.typeof_parameter(ansatz), length(selected_indices))

    else
        # MAKE SELECTION - GREEDY
        @info "Performing selection using greedy method"

        candidates = Dict(pool .=> scores)
        imap = Dict(pool .=> eachindex(pool))
        ops_to_add = Int64[]

        # remove candidate operators with scores below some threshold
        filter!(p-> p.second >= adapt_type.gradient_threshold, candidates)

        while !isempty(candidates)
            largest_score, G = findmax(candidates)
            G_support = support(G)
            filter!(p-> isdisjoint(support(p.first), G_support), candidates)
            push!(ops_to_add, imap[G])
        end

        selected_indices = ops_to_add
        selected_scores = scores[selected_indices];
        selected_generators = pool[selected_indices];
        selected_parameters = zeros(ADAPT.typeof_parameter(ansatz), length(ops_to_add));
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

# Returns the absolute value of the partial derivative of the observable with respect to the generator
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