#= DISCLAIMER: This file is meant to be included in ADAPT.jl.
    Names not defined in this file or in base Julia can be found there.
=#

import Random

"""
    evolve_state(
        ansatz::AbstractAnsatz,
        reference::QuantumState,
    )

Calculate the quantum state resulting from applying an ansatz to a given reference state.

"""
function evolve_state(
    ansatz::AbstractAnsatz,
    reference::QuantumState,
)
    state = deepcopy(reference)
    evolve_state!(ansatz, state)
    return state
end

"""
    evolve_state(
        G::Generator,
        θ::Parameter,
        ψ::QuantumState,
    )

Calculate the quantum state rotating `ψ` by an amount `x` about the axis defined by `G`.

"""
function evolve_state(
    G::Generator,
    θ::Parameter,
    Ψ::QuantumState,
)
    state = deepcopy(Ψ)
    evolve_state!(G, θ, state)
    return state
end

"""
    evolve_state!(
        ansatz::AbstractAnsatz,
        state::QuantumState,
    )

Apply an ansatz to the given quantum state, mutating and returning the state.

By default, generators with a lower index are applied to the state earlier.
This means that the equation for |Ψ⟩ would list out generators in reverse order.
Specific implementations of ansatze may override this behavior.

"""
function evolve_state!(
    ansatz::AbstractAnsatz,
    state::QuantumState,
)::QuantumState
    for (generator, parameter) in ansatz
        evolve_state!(generator, parameter, state)
    end
    return state
end

"""
    evolve_state!(
        G::Generator,
        θ::Parameter,
        ψ::QuantumState,
    )

Rotate a quantum state `ψ` by an amount `x` about the axis defined by `G`,
    mutating and returning `ψ`.

# Implementation

Typically, the "rotation" is the unitary operator `exp(-iθG)`,
    but different `Generator` types could have different effects.

"""
evolve_state!(
    ::Generator,
    ::Parameter,
    ::QuantumState,
) = NotImplementedError()




"""
    evaluate(
        ansatz::AbstractAnsatz,
        H::Observable,
        ψ0::QuantumState,
    )

Evaluate a cost-function with respect to a particular ADAPT state.

# Parameters
- `ansatz`: the ADAPT state
- `H`: the object defining the cost-function
- `ψ0`: an initial quantum state which the `ansatz` operates on

# Returns
- `energy`: a scalar number, whose type is `typeof_energy(observable)`

"""
function evaluate(
    ansatz::AbstractAnsatz,
    observable::Observable,
    reference::QuantumState,
)
    state = evolve_state(ansatz, reference)
    return evaluate(observable, state)
end

"""
    evaluate(
        H::Observable,
        Ψ::QuantumState,
    )

Evaluate a cost-function with respect to a particular quantum state.

# Parameters
- `H`: the object defining the cost-function
- `Ψ`: the quantum state

# Returns
- `energy`: a scalar number, whose type is `typeof_energy(observable)`

# Implementation

Typically, the "cost-function" is the expectation value `⟨Ψ|H|Ψ⟩`,
    but different `Observable` types could have different definitions.

"""
evaluate(
    ::Observable,
    ::QuantumState,
) = NotImplementedError()


"""
    partial(
        index::Int,
        ansatz::AbstractAnsatz,
        observable::Observable,
        reference::QuantumState,
    )

The partial derivative of a cost-function with respect to the i-th parameter in an ansatz.

# Parameters
- `index`: the index of the parameter to calculate within `ansatz`
- `ansatz`: the ADAPT state
- `observable`: the object defining the cost-function
- `reference`: an initial quantum state which the `ansatz` operates on

# Returns
- a number of type `typeof_energy(observable)`.

# Implementation

Typically, generators apply a unitary rotation,
    so the partial consists of a partial evolution up to the indexed generator,
    then a "kick" from the generator itself, then a final evolution,
    and a braket with the observable.
But, different ansatze may have a different procedure.

"""
function partial(
    index::Int,
    ansatz::AbstractAnsatz,
    observable::Observable,
    reference::QuantumState,
)
    return ADAPT.gradient(ansatz, observable, reference)[index]
end

"""
    gradient(
        ansatz::AbstractAnsatz,
        observable::Observable,
        reference::QuantumState,
    )

Construct a vector of partial derivatives with respect to each parameter in the ansatz.

# Parameters
- `ansatz`: the ADAPT state
- `H`: the object defining the cost-function
- `ψ0`: an initial quantum state which the `ansatz` operates on

# Returns
- a vector of type `typeof_energy(observable)`.

"""
function gradient(
    ansatz::AbstractAnsatz,
    observable::Observable,
    reference::QuantumState,
)
    F = typeof_energy(observable)
    result = Vector{F}(undef, length(ansatz))
    result = gradient!(result, ansatz, observable, reference)
    return result
end

"""
    gradient!(
        result::AbstractVector,
        ansatz::AbstractAnsatz,
        observable::Observable,
        reference::QuantumState,
    )

Fill a vector of partial derivatives with respect to each parameter in the ansatz.

# Parameters
- `result`: vector which will contain the gradient after calling this function
- `ansatz`: the ADAPT state
- `H`: the object defining the cost-function
- `ψ0`: an initial quantum state which the `ansatz` operates on

# Returns
- `result`

"""
function gradient!(
    result::AbstractVector,
    ansatz::AbstractAnsatz,
    observable::Observable,
    reference::QuantumState,
)
    cfd = FiniteDifferences.central_fdm(5, 1)
    x0 = copy(ADAPT.angles(ansatz))
    fn = make_costfunction(ansatz, observable, reference)
    result .= FiniteDifferences.grad(cfd, fn, x0)[1]
    return result
end



"""
    make_costfunction(
        ansatz::ADAPT.AbstractAnsatz,
        observable::ADAPT.Observable,
        reference::ADAPT.QuantumState,
    )

Construct a single-parameter cost-function f(x), where x is a parameter vector.

Note that calling f does *not* change the state of the ansatz
    (although actually it does temporarily, so this function is not thread-safe).

# Parameters
- `ansatz`: the ADAPT state
- `observable`: the object defining the cost-function
- `reference`: an initial quantum state which the `ansatz` operates on

# Returns
- `fn` a callable function `f(x)` where `x` is a vector of angles compatible with `ansatz`

"""
function make_costfunction(
    ansatz::ADAPT.AbstractAnsatz,
    observable::ADAPT.Observable,
    reference::ADAPT.QuantumState,
)
    x0 = copy(ADAPT.angles(ansatz))
    return (x) -> (
        x0 .= ADAPT.angles(ansatz);         # SAVE THE ORIGINAL STATE
        ADAPT.bind!(ansatz, x);
        f = evaluate(ansatz, observable, reference);
        ADAPT.bind!(ansatz, x0);            # RESTORE THE ORIGINAL STATE
        f
    )
end

"""
    make_gradfunction(
        ansatz::ADAPT.AbstractAnsatz,
        observable::ADAPT.Observable,
        reference::ADAPT.QuantumState,
    )

Construct a single-parameter gradient function g(x), where x is a parameter vector.

Note that calling g does *not* change the state of the ansatz
    (although actually it does temporarily, so this function is not thread-safe).

# Parameters
- `ansatz`: the ADAPT state
- `observable`: the object defining the cost-function
- `reference`: an initial quantum state which the `ansatz` operates on

# Returns
- `gd` a callable function `gd(x)` where `x` is a vector of angles compatible with `ansatz`

"""
function make_gradfunction(
    ansatz::ADAPT.AbstractAnsatz,
    observable::ADAPT.Observable,
    reference::ADAPT.QuantumState,
)
    g! = make_gradfunction!(ansatz, observable, reference)
    return (x) -> (
        ∇f = Vector{ADAPT.typeof_energy(observable)}(undef, length(x));
        g!(∇f, x);
        ∇f
    )
end

"""
    make_gradfunction!(
        ansatz::ADAPT.AbstractAnsatz,
        observable::ADAPT.Observable,
        reference::ADAPT.QuantumState,
    )

Construct a mutating gradient function g!(∇f, x), where x is a parameter vector.

Using this in place of `make_gradfunction` for optimization
    will tend to significantly reduce memory allocations.

Note that calling g! does *not* change the state of the ansatz
    (although actually it does temporarily, so this function is not thread-safe).

# Parameters
- `ansatz`: the ADAPT state
- `observable`: the object defining the cost-function
- `reference`: an initial quantum state which the `ansatz` operates on

# Returns
- `g!` a callable function `g!(∇f,x)`
  - `∇f` and `x` are vectors of angles compatible with `ansatz`.
    The first argument `∇f` is used to store the result; its initial values are ignored.

"""
function make_gradfunction!(
    ansatz::ADAPT.AbstractAnsatz,
    observable::ADAPT.Observable,
    reference::ADAPT.QuantumState,
)
    x0 = copy(ADAPT.angles(ansatz))
    return (∇f, x) -> (
        x0 .= ADAPT.angles(ansatz);         # SAVE THE ORIGINAL STATE
        ADAPT.bind!(ansatz, x);
        gradient!(∇f, ansatz, observable, reference);
        ADAPT.bind!(ansatz, x0);            # RESTORE THE ORIGINAL STATE
        ∇f
    )
end


"""
    sample_from_state(state, n_samples=1024)
Sample bitstrings from the final quantum state.

Parameters:
    state: A quantum state vector (can be real or complex numbers).
    n_samples: The number of bitstrings to sample.

Returns:
    A BitMatrix with rows = qubits and columns = samples.

"""
function sample_from_state(state::AbstractVector{<:Number}, n_samples::Int=1024)::BitMatrix
    probabilities = abs2.(state)
    n_states = length(state)

    if n_states < 2
        error("Invalid quantum state: state vector must have at least 2 elements")
    end

    if !ispow2(n_states)
        error("Invalid quantum state: length must be a power of 2, got $n_states")
    end

    n_qubits = Int(log2(n_states))
    
    # use the cumulative distribution function to sample the bitstrings
    cdf = cumsum(probabilities)

    # Ensure CDF ends at 1.0 (or close) for edge case handling
    if cdf[end] < 1.0 - 1e-10
        error("Probabilities don't sum to 1.0: sum=$(cdf[end])")
    end

    # 3. Pre-allocate BitMatrix: (Rows = Qubits, Cols = Samples)
    # Julia is Column-Major, so we store each sample as a column.
    bitstrings = BitMatrix(undef, n_qubits, n_samples)

    for i in 1:n_samples
        # Generate random number (uniformly sampled in [0, 1]) and find corresponding state
        r = rand()
        
        # find the first index of the cumulative distribution function 
        # that is greater than the uniformly sampled random number
        # uses binary search to find the index, i.e. O(log(2^n_qubits)) time
        selected_idx = searchsortedfirst(cdf, r)
        # Handle edge case where r is >= cdf[end] due to floating point imprecision, 
        #subtract 1 to get the correct index as Julia is 1-indexed
        selected_idx = clamp(selected_idx, 1, n_states) - 1
        
        # Convert index to bitstring
        for j in 1:n_qubits
            @inbounds bitstrings[j, i] = (selected_idx >> (j-1)) & 1
        end
    end
    
    return bitstrings
end

