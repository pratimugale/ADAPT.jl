using ADAPT
using Test

import PauliOperators: Pauli, ScaledPauli, ScaledPauliVector, PauliSum
import PauliOperators: SparseKetBasis, KetBitString

##########################################################################################
#= DEFINE RE-USABLE ADAPT OBJECTS =#

N = 4

BFGS = ADAPT.OptimOptimizer(:BFGS, g_tol=1e-3)

pools = (
    PauliSum = [
        [PauliSum(Pauli(N; Y=q)) for q in 1:N];
        [Pauli(N; X=[q,q+1]) + Pauli(N; Y=[q,q+1]) for q in 1:N-1];
    ],
    ScaledPauliVector = [
        [[1.0 * Pauli(N; Y=q)] for q in 1:N];
        [[1.0 * Pauli(N; X=[q,q+1]), 1.0 * Pauli(N; Y=[q,q+1])] for q in 1:N-1];
    ],
    # TODO: ScaledPauli
    # TODO: Pauli
)

observables = (
    PauliSum = let
        H = PauliSum(N)
        for q in 1:N-1
            sum!(H, Pauli(N; X=q))
            sum!(H, Pauli(N; Z=[q,q+1]))
        end
        sum!(H, Pauli(N; X=N))
        H
    end,
    ScaledPauliVector = let
        [1.0 * Pauli(N; Z=[q,q+1]) for q in 1:N-1]
    end,
    # TODO: ScaledPauli
    # TODO: Pauli
    Infidelity = let
        ψ = zeros(ComplexF64, 1<<N)
        ψ[4] = 1/√2
        ψ[13] = 1/√2
        ADAPT.OverlapADAPT.Infidelity(ψ)
    end,
)

references = (
    SparseKetBasis = let
        ket = KetBitString{N}(parse(Int128, "1100", base=2))
        sparseket = SparseKetBasis{N,ComplexF64}(ket => 1)
        sparseket
    end,
    Vector = let
        ψ = zeros(ComplexF64, 1<<N)
        ψ[13] = 1;
        ψ
    end,
)

##########################################################################################
#= FUNCTION TO RUN A VALIDATION, FOR A SPECIFIC COMBO =#

function run_tests(combo)
    label = join(map(string, combo), ".")

    ansatztype = ansatze[combo[1]]
    adapt = adapts[combo[2]]
    vqe = vqes[combo[3]]
    pool = pools[combo[4]]
    observable = observables[combo[5]]
    reference = references[combo[6]]

    ansatz = ansatztype(Float64, pool)

    ADAPT.validate(ansatz, adapt, vqe, pool, observable, reference; label=label)
end


##########################################################################################
#= VALIDATE SELECT COMBINATIONS =#

@testset "ADAPT.jl" verbose=true begin
    #= Run tests from other files =#
    include("max3sat_exact_hamiltonian.jl")
    include("max3sat_approx_hamiltonian.jl")
    include("max3sat_bit_flip_symmetry.jl")
    include("check_pool_completeness.jl")

    @testset "Basics" begin
        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:ScaledPauliVector]),
            ADAPT.VANILLA,
            BFGS,
            pools[:ScaledPauliVector],
            observables[:PauliSum],
            references[:Vector];
            label = "ScaledPauli[] Pool, Statevector",
        )

        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:ScaledPauliVector]),
            ADAPT.VANILLA,
            BFGS,
            pools[:ScaledPauliVector],
            observables[:PauliSum],
            references[:SparseKetBasis];
            label = "ScaledPauli[] Pool, SparseKetBasis",
        )

        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:PauliSum]),
            ADAPT.VANILLA,
            BFGS,
            pools[:PauliSum],
            observables[:PauliSum],
            references[:Vector];
            label = "PauliSum Pool, Statevector",
        )

        # ADAPT.validate(
        #     ADAPT.Ansatz(Float64, pools[:PauliSum]),
        #     ADAPT.VANILLA,
        #     BFGS,
        #     pools[:PauliSum],
        #     observables[:PauliSum],
        #     references[:SparseKetBasis];
        #     label = "PauliSum Pool, SparseKetBasis",
        # )
        # TODO: Just some linear algebra, but SparseKetBasis is changing drastically soon.
    end

    @testset "sample_from_state" begin
        import Random
        Random.seed!(42)  # For reproducibility

        # Test 1: Deterministic state |00⟩ (2 qubits)
        ψ_00 = zeros(ComplexF64, 4)
        ψ_00[1] = 1.0  # |00⟩ is index 1
        samples = ADAPT.sample_from_state(ψ_00, 100)
        @test size(samples) == (2, 100)
        @test all(samples .== false)  # All bits should be 0

        # Test 2: Deterministic state |11⟩ (2 qubits)
        ψ_11 = zeros(ComplexF64, 4)
        ψ_11[4] = 1.0  # |11⟩ is index 4 (binary 11 = 3, 0-indexed, so Julia index 4)
        samples = ADAPT.sample_from_state(ψ_11, 100)
        @test size(samples) == (2, 100)
        @test all(samples .== true)  # All bits should be 1

        # Test 3: Deterministic state |101⟩ (3 qubits)
        # |101⟩ = 1*2^0 + 0*2^1 + 1*2^2 = 1 + 4 = 5, so Julia index 6
        ψ_101 = zeros(ComplexF64, 8)
        ψ_101[6] = 1.0
        samples = ADAPT.sample_from_state(ψ_101, 50)
        @test size(samples) == (3, 50)
        expected = [true, false, true]  # bits: q1=1, q2=0, q3=1
        for i in 1:50
            @test samples[:, i] == expected
        end

        # Test 4: Equal superposition (2 qubits) - should get all 4 states
        ψ_equal = ones(ComplexF64, 4) / 2.0  # |ψ|² = 1/4 for each
        samples = ADAPT.sample_from_state(ψ_equal, 10000)
        @test size(samples) == (2, 10000)
        # Count occurrences of each state
        counts = Dict{Vector{Bool}, Int}()
        for i in 1:10000
            bitstring = Vector{Bool}(samples[:, i])
            counts[bitstring] = get(counts, bitstring, 0) + 1
        end
        # All 4 states should appear (with high probability given 10000 samples)
        @test length(counts) == 4
        # Each state should appear roughly 2500 times (±200 for statistical noise)
        for (_, count) in counts
            @test 2300 < count < 2700
        end

        # Test 5: Biased state - |0⟩ has 90% probability, |1⟩ has 10%
        ψ_biased = zeros(ComplexF64, 2)
        ψ_biased[1] = sqrt(0.9)  # |0⟩
        ψ_biased[2] = sqrt(0.1)  # |1⟩
        samples = ADAPT.sample_from_state(ψ_biased, 10000)
        @test size(samples) == (1, 10000)
        n_zeros = count(x -> !x, samples[1, :])
        # Should be roughly 9000 zeros (±200 for statistical noise)
        @test 8800 < n_zeros < 9200

        # Test 6: Error case - non-power-of-2 length
        @test_throws ErrorException ADAPT.sample_from_state([1.0, 0.0, 0.0], 10)

        # Test 7: Error case - single element (< 2)
        @test_throws ErrorException ADAPT.sample_from_state([1.0], 10)
    end

    @testset "Optimization-Free" begin
        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:ScaledPauliVector]),
            ADAPT.VANILLA,
            ADAPT.OptimizationFreeADAPT.OPTIMIZATION_FREE,
            pools[:ScaledPauliVector],
            observables[:PauliSum],
            references[:Vector];
            label = "ScaledPauli[] Pool, Statevector",
        )
    end

    @testset "Degenerate-ADAPT" begin
        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:ScaledPauliVector]),
            ADAPT.Degenerate_ADAPT.DegenerateADAPT(1e-4),
            BFGS,
            pools[:ScaledPauliVector],
            observables[:PauliSum],
            references[:Vector];
            label = "ScaledPauli[] Pool, Statevector",
        )
    end

    @testset "TETRIS-ADAPT" begin
        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:ScaledPauliVector]),
            ADAPT.TETRIS_ADAPT.TETRISADAPT(1e-4),
            BFGS,
            pools[:ScaledPauliVector],
            observables[:PauliSum],
            references[:Vector];
            label = "ScaledPauli[] Pool, Statevector",
        )
    end

    @testset "ADAPT-QAOA" begin
        ADAPT.validate(
            ADAPT.ADAPT_QAOA.QAOAAnsatz(0.1, observables[:ScaledPauliVector]),
            ADAPT.VANILLA,
            BFGS,
            pools[:ScaledPauliVector],
            observables[:ScaledPauliVector],
            references[:Vector];
            label = "ScaledPauli[] Pool, Statevector",
            scores = nothing,   # Default validation can't handle new ansatz indexing.
        )

        qaoa_H = ADAPT.ADAPT_QAOA.QAOAObservable(observables[:ScaledPauliVector])
        ADAPT.validate(
            ADAPT.ADAPT_QAOA.DiagonalQAOAAnsatz(0.1, pools[:ScaledPauliVector], qaoa_H),
            ADAPT.VANILLA,
            BFGS,
            pools[:ScaledPauliVector],
            observables[:ScaledPauliVector],
            references[:Vector];
            label = "DiagonalQAOAAnsatz, ScaledPauli[] Pool, Statevector",
            scores = nothing,   # Default validation can't handle new ansatz indexing.
        )
        # TODO: Once we have strict versioning, DiagonalQAOAAnsatz will replace QAOAAnsatz.
    end

    @testset "Overlap" begin
        overlap = let
            ψ = zeros(ComplexF64, 1<<N)
            ψ[4] = 1/√2
            ψ[13] = 1/√2
            ADAPT.OverlapADAPT.Infidelity(ψ)
        end

        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:ScaledPauliVector]),
            ADAPT.VANILLA,
            BFGS,
            pools[:ScaledPauliVector],
            overlap,
            references[:Vector];
            label = "Statevector",
        )

        ADAPT.validate(
            ADAPT.Ansatz(Float64, pools[:ScaledPauliVector]),
            ADAPT.VANILLA,
            BFGS,
            pools[:ScaledPauliVector],
            overlap,
            references[:SparseKetBasis];
            label = "SparseKetBasis",
        )
    end
end
