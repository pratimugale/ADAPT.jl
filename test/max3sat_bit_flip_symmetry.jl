using ADAPT
using Test
using LinearAlgebra
using Random

const Literal = ADAPT.Hamiltonians.Max3SAT.Types.Literal
const Clause = ADAPT.Hamiltonians.Max3SAT.Types.Clause
const Formula = ADAPT.Hamiltonians.Max3SAT.Types.Formula
const get_approximate_hamiltonian = ADAPT.Hamiltonians.Max3SAT.get_approximate_hamiltonian
const get_exact_hamiltonian = ADAPT.Hamiltonians.Max3SAT.get_exact_hamiltonian

@testset "Max3SAT Bit Flip Symmetry" begin
    Random.seed!(1234)

    n_instances = 10
    println("Running test on $n_instances random instances...")

    for i in 1:n_instances
        n_vars = 10 # Small enough number of variables for full matrix simulation
        n_clauses = 20

        # Generate random formula
        clauses = Clause[]
        for _ in 1:n_clauses
            # Random literals
            l1 = Literal(rand(1:n_vars), rand(Bool))
            l2 = Literal(rand(1:n_vars), rand(Bool))
            l3 = Literal(rand(1:n_vars), rand(Bool))
            push!(clauses, Clause((l1, l2, l3)))
        end
        formula = Formula(clauses)

        function pretty_print_formula(f::Formula)
            str_clauses = String[]
            for clause in f.clauses
                lits = clause.lits
                str_lits = map(l -> (l.neg ? "¬x$(l.var)" : "x$(l.var)"), lits)
                push!(str_clauses, "(" * join(str_lits, " ∨ ") * ")")
            end
            return join(str_clauses, " ∧ ")
        end

        println("Formula: $(pretty_print_formula(formula))")

        hamiltonian_terms = get_approximate_hamiltonian(formula, n_vars)

        dim = 2^n_vars
        psi = rand(ComplexF64, dim)
        psi /= norm(psi)

        E1 = ADAPT.evaluate(hamiltonian_terms, psi)

        psi_flipped = reverse(psi)

        # Calculate energy E2 = <psi_flipped|H|psi_flipped>
        E2 = ADAPT.evaluate(hamiltonian_terms, psi_flipped)

        println("Instance $i: E1 = $E1, E2 = $E2, Diff = $(abs(E1 - E2))")

        # Check approximate equality 
        @test isapprox(E1, E2, atol=1e-15)

        # Check exact Hamiltonian (should NOT be symmetric)
        exact_hamiltonian_terms = get_exact_hamiltonian(formula, n_vars)
        E1_exact = ADAPT.evaluate(exact_hamiltonian_terms, psi)
        E2_exact = ADAPT.evaluate(exact_hamiltonian_terms, psi_flipped)
        
        println("Exact H: E1 = $E1_exact, E2 = $E2_exact, Diff = $(abs(E1_exact - E2_exact))")
        @test !isapprox(E1_exact, E2_exact, atol=1e-6)
    end
end
