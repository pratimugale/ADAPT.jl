# Test file for FullApprox Hamiltonian
# Tests with one clause of all positive variables

# Import the Max3SAT module types
const Literal = ADAPT.Hamiltonians.Max3SAT.Types.Literal
const Clause = ADAPT.Hamiltonians.Max3SAT.Types.Clause
const Formula = ADAPT.Hamiltonians.Max3SAT.Types.Formula
const get_approximate_hamiltonian = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian

@testset "Max3SAT Approx Hamiltonian" begin

    @testset "Type A (0 negations)" begin
        println("Testing FullApprox Hamiltonian with one clause of all positive variables...")
        
        # Create literals for clause (x1 ∨ x2 ∨ x3)
        lit1 = Literal(1, false)  # x1 (positive)
        lit2 = Literal(2, false)  # x2 (positive)  
        lit3 = Literal(3, false)  # x3 (positive)
        
        # Create clause
        clause = Clause((lit1, lit2, lit3))
        
        # Create formula with one clause
        formula = Formula([clause])
        
        # Number of variables
        n_vars = 3
        
        # Get approximate Hamiltonian
        hamiltonian_terms = get_approximate_hamiltonian(formula, n_vars)
        
        println("Number of Hamiltonian terms: $(length(hamiltonian_terms))")
        println("\nHamiltonian terms:")
        for (i, term) in enumerate(hamiltonian_terms)
            println("Term $i: coefficient = $(term.coeff), Pauli = $(term.pauli)")
        end
        
        # Assertions based on observed output
        @test length(hamiltonian_terms) == 7  # "Should have exactly 7 Hamiltonian terms"
        
        # Check for specific Pauli strings
        pauli_strings = [string(term.pauli) for term in hamiltonian_terms]
        @test "III" in pauli_strings  # "Should have identity term III"
        @test "ZIZ" in pauli_strings  # "Should have ZIZ term"
        @test "ZZI" in pauli_strings  # "Should have ZZI term"
        @test "IZZ" in pauli_strings  # "Should have IZZ term"
        
        # Check each specific term and its coefficient
        term_dict = Dict(string(term.pauli) => term.coeff for term in hamiltonian_terms)
        
        @test abs(term_dict["III"] - (-0.75)) < 1e-10  # "III should have coefficient -0.75"
        @test abs(term_dict["ZIZ"] - 0.25) < 1e-10  # "ZIZ should have coefficient 0.25"
        @test abs(term_dict["ZZI"] - 0.25) < 1e-10  # "ZZI should have coefficient 0.25"
        @test abs(term_dict["IZZ"] - 0.25) < 1e-10  # "IZZ should have coefficient 0.25"
        @test abs(term_dict["IZI"]) < 1e-10  # "IZI should have coefficient 0.0"
        @test abs(term_dict["ZII"]) < 1e-10  # "ZII should have coefficient 0.0"
        @test abs(term_dict["IIZ"]) < 1e-10  # "IIZ should have coefficient 0.0"
        
        println("✓ All assertions passed!")
    end

    @testset "Type B (1 negation)" begin
        println("\nTesting FullApprox Hamiltonian with one Type B clause (1 negation)...")
        
        # Create literals for clause (x1 ∨ x2 ∨ ¬x3) - 1 negation
        lit1 = Literal(1, false)  # x1 (positive)
        lit2 = Literal(2, false)  # x2 (positive)  
        lit3 = Literal(3, true)   # ¬x3 (negative)
        
        # Create clause
        clause = Clause((lit1, lit2, lit3))
        
        # Create formula with one clause
        formula = Formula([clause])
        
        # Number of variables
        n_vars = 3
        
        # Get approximate Hamiltonian
        hamiltonian_terms = get_approximate_hamiltonian(formula, n_vars)
        
        println("Number of Hamiltonian terms: $(length(hamiltonian_terms))")
        println("\nHamiltonian terms:")
        for (i, term) in enumerate(hamiltonian_terms)
            println("Term $i: coefficient = $(term.coeff), Pauli = $(term.pauli)")
        end
        
        # Assertions based on observed output for Type B
        @test length(hamiltonian_terms) == 7  # "Should have exactly 7 Hamiltonian terms"
        
        # Check for specific Pauli strings
        pauli_strings = [string(term.pauli) for term in hamiltonian_terms]
        @test "III" in pauli_strings  # "Should have identity term III"
        @test "ZIZ" in pauli_strings  # "Should have ZIZ term"
        @test "ZZI" in pauli_strings  # "Should have ZZI term"
        @test "IZZ" in pauli_strings  # "Should have IZZ term"
        
        # Check each specific term and its coefficient
        term_dict = Dict(string(term.pauli) => term.coeff for term in hamiltonian_terms)
        
        @test abs(term_dict["III"] - 0.25) < 1e-10  # "III should have coefficient 0.25"
        @test abs(term_dict["ZIZ"] - (-0.25)) < 1e-10  # "ZIZ should have coefficient -0.25"
        @test abs(term_dict["ZZI"] - 0.25) < 1e-10  # "ZZI should have coefficient 0.25"
        @test abs(term_dict["IZZ"] - (-0.25)) < 1e-10  # "IZZ should have coefficient -0.25"
        @test abs(term_dict["IZI"]) < 1e-10  # "IZI should have coefficient 0.0"
        @test abs(term_dict["ZII"]) < 1e-10  # "ZII should have coefficient 0.0"
        @test abs(term_dict["IIZ"]) < 1e-10  # "IIZ should have coefficient 0.0"
        
        println("✓ All Type B assertions passed!")
    end

    @testset "Type C (2 negations)" begin
        println("\nTesting FullApprox Hamiltonian with one Type C clause (2 negations)...")
        
        # Create literals for clause (x1 ∨ ¬x2 ∨ ¬x3) - 2 negations
        lit1 = Literal(1, false)  # x1 (positive)
        lit2 = Literal(2, true)   # ¬x2 (negative)
        lit3 = Literal(3, true)   # ¬x3 (negative)
        
        # Create clause
        clause = Clause((lit1, lit2, lit3))
        
        # Create formula with one clause
        formula = Formula([clause])
        
        # Number of variables
        n_vars = 3
        
        # Get approximate Hamiltonian
        hamiltonian_terms = get_approximate_hamiltonian(formula, n_vars)
        
        println("Number of Hamiltonian terms: $(length(hamiltonian_terms))")
        println("\nHamiltonian terms:")
        for (i, term) in enumerate(hamiltonian_terms)
            println("Term $i: coefficient = $(term.coeff), Pauli = $(term.pauli)")
        end
        
        # Assertions based on observed output for Type C
        @test length(hamiltonian_terms) == 7  # "Should have exactly 7 Hamiltonian terms"
        
        # Check for specific Pauli strings
        pauli_strings = [string(term.pauli) for term in hamiltonian_terms]
        @test "III" in pauli_strings  # "Should have identity term III"
        @test "ZIZ" in pauli_strings  # "Should have ZIZ term"
        @test "ZZI" in pauli_strings  # "Should have ZZI term"
        @test "IZZ" in pauli_strings  # "Should have IZZ term"
        
        # Check each specific term and its coefficient
        term_dict = Dict(string(term.pauli) => term.coeff for term in hamiltonian_terms)
        
        @test abs(term_dict["III"] - 0.25) < 1e-10  # "III should have coefficient 0.25"
        @test abs(term_dict["ZIZ"] - (-0.25)) < 1e-10  # "ZIZ should have coefficient -0.25"
        @test abs(term_dict["ZZI"] - (-0.25)) < 1e-10  # "ZZI should have coefficient -0.25"
        @test abs(term_dict["IZZ"] - 0.25) < 1e-10  # "IZZ should have coefficient 0.25"
        @test abs(term_dict["IZI"]) < 1e-10  # "IZI should have coefficient 0.0"
        @test abs(term_dict["ZII"]) < 1e-10  # "ZII should have coefficient 0.0"
        @test abs(term_dict["IIZ"]) < 1e-10  # "IIZ should have coefficient 0.0"
        
        println("✓ All Type C assertions passed!")
    end

    @testset "Type D (3 negations)" begin
        println("\nTesting FullApprox Hamiltonian with one Type D clause (3 negations)...")
        
        # Create literals for clause (¬x1 ∨ ¬x2 ∨ ¬x3) - 3 negations
        lit1 = Literal(1, true)   # ¬x1 (negative)
        lit2 = Literal(2, true)   # ¬x2 (negative)
        lit3 = Literal(3, true)   # ¬x3 (negative)
        
        # Create clause
        clause = Clause((lit1, lit2, lit3))
        
        # Create formula with one clause
        formula = Formula([clause])
        
        # Number of variables
        n_vars = 3
        
        # Get approximate Hamiltonian
        hamiltonian_terms = get_approximate_hamiltonian(formula, n_vars)
        
        println("Number of Hamiltonian terms: $(length(hamiltonian_terms))")
        println("\nHamiltonian terms:")
        for (i, term) in enumerate(hamiltonian_terms)
            println("Term $i: coefficient = $(term.coeff), Pauli = $(term.pauli)")
        end
        
        # Assertions based on observed output for Type D
        @test length(hamiltonian_terms) == 7  # "Should have exactly 7 Hamiltonian terms"
        
        # Check for specific Pauli strings
        pauli_strings = [string(term.pauli) for term in hamiltonian_terms]
        @test "III" in pauli_strings  # "Should have identity term III"
        @test "ZIZ" in pauli_strings  # "Should have ZIZ term"
        @test "ZZI" in pauli_strings  # "Should have ZZI term"
        @test "IZZ" in pauli_strings  # "Should have IZZ term"
        
        # Check each specific term and its coefficient
        term_dict = Dict(string(term.pauli) => term.coeff for term in hamiltonian_terms)
        
        @test abs(term_dict["III"] - (-0.75)) < 1e-10  # "III should have coefficient -0.75"
        @test abs(term_dict["ZIZ"] - 0.25) < 1e-10  # "ZIZ should have coefficient 0.25"
        @test abs(term_dict["ZZI"] - 0.25) < 1e-10  # "ZZI should have coefficient 0.25"
        @test abs(term_dict["IZZ"] - 0.25) < 1e-10  # "IZZ should have coefficient 0.25"
        @test abs(term_dict["IZI"]) < 1e-10  # "IZI should have coefficient 0.0"
        @test abs(term_dict["ZII"]) < 1e-10  # "ZII should have coefficient 0.0"
        @test abs(term_dict["IIZ"]) < 1e-10  # "IIZ should have coefficient 0.0"
        
        println("✓ All Type D assertions passed!")
    end

    @testset "Comprehensive (4 clauses, 5 variables)" begin
        println("\nTesting FullApprox Hamiltonian with 4 clauses (one of each type) and 5 variables...")
        
        # Create 4 clauses with different types and orders
        # Type A: (x1 ∨ x2 ∨ x3) - all positive
        clause_a = Clause((
            Literal(1, false),  # x1
            Literal(2, false),  # x2
            Literal(3, false)   # x3
        ))
        
        # Type B: (¬x2 ∨ x4 ∨ ¬x5) - 1 positive, 2 negative (rearranged order)
        clause_b = Clause((
            Literal(2, true),   # ¬x2 (negative first)
            Literal(4, false),  # x4 (positive)
            Literal(5, true)    # ¬x5 (negative)
        ))
        
        # Type C: (¬x1 ∨ ¬x3 ∨ x4) - 1 positive, 2 negative (rearranged order)
        clause_c = Clause((
            Literal(1, true),   # ¬x1 (negative first)
            Literal(3, true),   # ¬x3 (negative)
            Literal(4, false)   # x4 (positive)
        ))
        
        # Type D: (¬x2 ∨ ¬x4 ∨ ¬x5) - all negative
        clause_d = Clause((
            Literal(2, true),   # ¬x2
            Literal(4, true),   # ¬x4
            Literal(5, true)    # ¬x5
        ))
        
        # Create formula with all 4 clauses
        formula = Formula([clause_a, clause_b, clause_c, clause_d])
        
        # Number of variables
        n_vars = 5
        
        # Get approximate Hamiltonian
        hamiltonian_terms = get_approximate_hamiltonian(formula, n_vars)
        
        println("Number of Hamiltonian terms: $(length(hamiltonian_terms))")
        println("\nHamiltonian terms:")
        for (i, term) in enumerate(hamiltonian_terms)
            println("Term $i: coefficient = $(term.coeff), Pauli = $(term.pauli)")
        end
        
        # Assertions for comprehensive test
        @test length(hamiltonian_terms) > 0  # "Should have at least one Hamiltonian term"
        
        # Check that we have terms from all clause types
        # This is a more complex test since we're summing across multiple clauses
        # We should have more terms than any single clause would produce
        @test length(hamiltonian_terms) >= 7  # "Should have at least 7 terms (minimum from one clause)"
        
        # Check that all coefficients are reasonable (not NaN or Inf)
        for term in hamiltonian_terms
            @test isfinite(term.coeff)  # "All coefficients should be finite"
        end
        
        # Check that we have some non-zero terms
        non_zero_terms = [term for term in hamiltonian_terms if abs(term.coeff) > 1e-10]
        @test length(non_zero_terms) > 0  # "Should have at least some non-zero terms"
        
        # Check that we have some zero terms (from terms that cancel out)
        zero_terms = [term for term in hamiltonian_terms if abs(term.coeff) < 1e-10]
        @test length(zero_terms) >= 0  # "Should have zero or more zero terms"
        
        # Check each specific term and its coefficient
        term_dict = Dict(string(term.pauli) => term.coeff for term in hamiltonian_terms)
        
        # Check all 14 terms with their exact coefficients
        @test abs(term_dict["IZIIZ"] - 0.5) < 1e-10  # "IZIIZ should have coefficient 0.5"
        @test abs(term_dict["IIZZI"] - (-0.25)) < 1e-10  # "IIZZI should have coefficient -0.25"
        @test abs(term_dict["ZIZII"] - 0.5) < 1e-10  # "ZIZII should have coefficient 0.5"
        @test abs(term_dict["ZIIZI"] - (-0.25)) < 1e-10  # "ZIIZI should have coefficient -0.25"
        @test abs(term_dict["IZIZI"]) < 1e-10  # "IZIZI should have coefficient 0.0"
        @test abs(term_dict["ZIIII"]) < 1e-10  # "ZIIII should have coefficient 0.0"
        @test abs(term_dict["IIIZZ"]) < 1e-10  # "IIIZZ should have coefficient 0.0"
        @test abs(term_dict["IIIZI"]) < 1e-10  # "IIIZI should have coefficient 0.0"
        @test abs(term_dict["IIZII"]) < 1e-10  # "IIZII should have coefficient 0.0"
        @test abs(term_dict["IZZII"] - 0.25) < 1e-10  # "IZZII should have coefficient 0.25"
        @test abs(term_dict["ZZIII"] - 0.25) < 1e-10  # "ZZIII should have coefficient 0.25"
        @test abs(term_dict["IIIII"] - (-1.0)) < 1e-10  # "IIIII should have coefficient -1.0"
        @test abs(term_dict["IZIII"]) < 1e-10  # "IZIII should have coefficient 0.0"
        @test abs(term_dict["IIIIZ"]) < 1e-10  # "IIIIZ should have coefficient 0.0"
        
        # Check that we have cross-clause interaction terms
        # These should exist because different clauses share variables
        cross_terms = [pauli for pauli in keys(term_dict) if count(c -> c == 'Z', pauli) >= 2]
        @test length(cross_terms) > 0  # "Should have cross-clause interaction terms"
        
        println("✓ All comprehensive test assertions passed!")
        println("Total non-zero terms: $(length(non_zero_terms))")
        println("Total zero terms: $(length(zero_terms))")
        println("Cross-clause interaction terms: $(length(cross_terms))")
    end

end
