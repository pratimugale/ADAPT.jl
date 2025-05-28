import ..ADAPT
import PauliOperators: ScaledPauliVector

"""
    DiagonalQAOAAnsatz{F<:Parameter,G<:Generator}(
        observable::QAOAObservable,
        γ0::F,
        generators::Vector{G},
        β_parameters::Vector{F},
        γ_parameters::Vector{F},
        optimized::Bool,
        converged::Bool,
    )

An ADAPT state suitable for ADAPT-QAOA.
The standard ADAPT generators are interspersed with the observable itself.

# Type Parameters
- `F`: the number type for the parameters (usually `Float64` is appropriate).
- `G`: the generator type.

# Parameter
- `observable`: the observable, which is interspersed with generators when evolving
- `γ0`: initial coefficient of the observable, whenever a new generator is added
- `generators`: list of current generators (i.e. mixers)
- `β_parameters`: list of current generator coefficients
- `γ_parameters`: list of current observable coefficients
- `optimized`: whether the current parameters are flagged as optimal
- `converged`: whether the current generators are flagged as converged

"""
struct DiagonalQAOAAnsatz{F,G} <: ADAPT.AbstractAnsatz{F,G}
    observable::QAOAObservable
    γ0::F
    generators::Vector{G}
    β_parameters::Vector{F}
    γ_parameters::Vector{F}
    optimized::Ref{Bool}
    converged::Ref{Bool}
end

ADAPT.__get__generators(ansatz::DiagonalQAOAAnsatz) = vec(permutedims(
    hcat(fill(ansatz.observable, length(ansatz.generators)), ansatz.generators)
))
ADAPT.__get__parameters(ansatz::DiagonalQAOAAnsatz) = vec(permutedims(
    hcat(ansatz.γ_parameters, ansatz.β_parameters)
))
    #= TODO: These functions are actually redundant.
        We should probably alter the `AbstractAnsatz` interface. =#

ADAPT.__get__optimized(ansatz::DiagonalQAOAAnsatz) = ansatz.optimized
ADAPT.__get__converged(ansatz::DiagonalQAOAAnsatz) = ansatz.converged

"""
    DiagonalQAOAAnsatz(γ0, pool, observable)

Convenience constructor for initializing an empty ansatz.

# Parameters
- γ0
- pool
- observable

Note that the observable must be a `QAOAObservable`.

"""
DiagonalQAOAAnsatz(γ0, pool, observable) = DiagonalQAOAAnsatz(
    observable, γ0,
    eltype(pool)[],
    typeof(γ0)[],
    typeof(γ0)[],
    Ref(true), Ref(false),
)


##########################################################################################
#= AbstractVector interface. =#

#= Convenience function for extracting the half-index of integer `ix`. =#
# half(ix) = 1 + ((ix-1) >> 1)

Base.size(ansatz::DiagonalQAOAAnsatz) = size(ansatz.generators) .<< 1
Base.IndexStyle(::Type{<:DiagonalQAOAAnsatz}) = IndexLinear()

function Base.getindex(ansatz::DiagonalQAOAAnsatz, i::Int)
    ((i-1) & 1 == 0) && return ansatz.observable => ansatz.γ_parameters[half(i)]
    return ansatz.generators[half(i)] => ansatz.β_parameters[half(i)]
end

function Base.setindex!(ansatz::DiagonalQAOAAnsatz{F,G}, pair::Pair{G,F}, i::Int) where {F,G}
    #= TODO: We are making a major assumption,
            that setindex! is only called in the context of push!(G => x),
            i.e. attaching a new generator.
        Thus, we assume `ansatz[i] = (G => x)` is never called. =#
    ansatz.generators[half(i)] = pair.first
    ansatz.β_parameters[half(i)] = pair.second
    ansatz.γ_parameters[half(i)] = ansatz.γ0
end

function Base.resize!(ansatz::DiagonalQAOAAnsatz, nl::Int)
    resize!(ansatz.generators, half(nl))
    resize!(ansatz.β_parameters, half(nl))
    resize!(ansatz.γ_parameters, half(nl))
end

##########################################################################################
#= Evolution. =#

function ADAPT.angles(ansatz::DiagonalQAOAAnsatz)
    #= TODO: We should be able to make this a view, to avoid allocations.

    Seems we ought replace a whole bunch of `copy(angles...)` with `collect(angles...)`,
        but otherwise such an implementation should be fine.

    =#
    return vec(permutedims(hcat(ansatz.γ_parameters, ansatz.β_parameters)))
end

function ADAPT.bind!(ansatz::DiagonalQAOAAnsatz{F,G}, x::AbstractVector{F}) where {F,G}
    x = reshape(x, 2, :)
    ansatz.γ_parameters .= @view(x[1,:])
    ansatz.β_parameters .= @view(x[2,:])
end

##########################################################################################
#= Improved scoring for vanilla ADAPT. =#

#= TODO: There's a major dispatch ambiguity problem.

For now I'm hard-coding the generator/observable/protocol types,
    but this really shouldn't be necessary.

I think adding a `calculate_score(..., state)` into the core interface,
    so that `calculate_score(ansatz, ..., reference)` has an obvious dispatch
    (and so does `calculate_scores(...)`, for that matter!)
    may help, but I'm not sure it solves it.

Need to think it through a bit more.

But ultimately we shouldn't need to import PauliOperators here.

=#

import ADAPT.Basics.MyPauliOperators
import ADAPT.Basics.MyPauliOperators: Pauli, ScaledPauli, PauliSum, ScaledPauliVector
AnyPauli = Union{Pauli, ScaledPauli, PauliSum, ScaledPauliVector}
# TODO: Replace `MyPauliOperators` with `PauliOperators` throughout, once merged.

function ADAPT.calculate_score(
    ansatz::DiagonalQAOAAnsatz,
    ::ADAPT.Basics.VanillaADAPT,
    generator::AnyPauli,
    observable::AnyPauli,
    reference::ADAPT.QuantumState,
)
    state = ADAPT.evolve_state(ansatz, reference)
    ADAPT.evolve_state!(ansatz.observable, ansatz.γ0, state)
    return abs(ADAPT.Basics.MyPauliOperators.measure_commutator(
            generator, observable, state))
end

function ADAPT.calculate_score(
    ansatz::DiagonalQAOAAnsatz,
    ::ADAPT.Degenerate_ADAPT.DegenerateADAPT,
    generator::AnyPauli,
    observable::AnyPauli,
    reference::ADAPT.QuantumState,
)
    state = ADAPT.evolve_state(ansatz, reference)
    ADAPT.evolve_state!(ansatz.observable, ansatz.γ0, state)
    return abs(ADAPT.Basics.MyPauliOperators.measure_commutator(
            generator, observable, state))
end