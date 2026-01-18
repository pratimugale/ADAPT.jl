@testset "Hamiltonians" begin
    @testset "Max3SAT" begin
        # Import the Max3SAT module types
        Literal = ADAPT.Hamiltonians.Max3SAT.Types.Literal
        Clause = ADAPT.Hamiltonians.Max3SAT.Types.Clause
        Formula = ADAPT.Hamiltonians.Max3SAT.Types.Formula
        get_exact_hamiltonian = ADAPT.Hamiltonians.Max3SAT.get_exact_hamiltonian

        # Test 1: Single clause (x1 OR x2 OR x3) - all positive literals
        # Satisfied by any assignment except (0,0,0)
        clause1 = Clause((Literal(1, false), Literal(2, false), Literal(3, false)))
        formula1 = Formula([clause1])
        H1 = get_exact_hamiltonian(formula1, 3)
        
        @test length(H1) > 0  # Should have multiple terms
        @test length(H1) <= 8  # At most 8 terms from expanding 3 literals
        
        # Test 2: Single clause with negations (NOT x1 OR x2 OR NOT x3)
        clause2 = Clause((Literal(1, true), Literal(2, false), Literal(3, true)))
        formula2 = Formula([clause2])
        H2 = get_exact_hamiltonian(formula2, 3)
        
        @test length(H2) > 0
        @test length(H2) <= 8
        
        # Test 3: Multiple clauses - check that Hamiltonian is built correctly
        clause3a = Clause((Literal(1, false), Literal(2, false), Literal(3, false)))
        clause3b = Clause((Literal(1, true), Literal(2, true), Literal(3, true)))
        formula3 = Formula([clause3a, clause3b])
        H3 = get_exact_hamiltonian(formula3, 3)
        
        @test length(H3) > 0  # Should have terms
        
        # Test 4: Verify energy for ALL 8 basis states
        # For clause (x1 OR x2 OR x3):
        #   - |000⟩ (index 1) → unsatisfied, energy = 1
        #   - All other states → satisfied, energy = 0
        for idx in 1:8
            ψ = zeros(ComplexF64, 8)
            ψ[idx] = 1.0
            energy = ADAPT.evaluate(H1, ψ)
            
            @test isa(energy, Real)
            
            if idx == 1  # |000⟩ - only unsatisfied state
                @test isapprox(energy, 1.0, atol=1e-10)
            else  # All other states satisfy (x1 OR x2 OR x3)
                @test isapprox(energy, 0.0, atol=1e-10)
            end
        end
        
        # Test 5: Verify energy for clause with negations (NOT x1 OR x2 OR NOT x3)
        # This clause is unsatisfied only when x1=1, x2=0, x3=1 → |101⟩ = index 6
        for idx in 1:8
            ψ = zeros(ComplexF64, 8)
            ψ[idx] = 1.0
            energy = ADAPT.evaluate(H2, ψ)
            
            if idx == 6  # |101⟩ - only unsatisfied state for (NOT x1 OR x2 OR NOT x3)
                @test isapprox(energy, 1.0, atol=1e-10)
            else
                @test isapprox(energy, 0.0, atol=1e-10)
            end
        end
    end
end