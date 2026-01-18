module Max3SAT
    
    import PauliOperators: Pauli, PauliSum, ScaledPauli

    export Types

    module Types

        "Literal: (variable index, negated flag)"
        struct Literal
            var::Int
            neg::Bool
        end

        "Clause: disjunction (OR) of 3 literals"
        struct Clause
            lits::NTuple{3, Literal}
        end

        "Formula: conjunction (AND) of multiple clauses"
        struct Formula
            clauses::Vector{Clause}
        end

        # Treat Formula like a collection of Clause
        Base.length(f::Formula) = length(f.clauses)
        Base.iterate(f::Formula, state...) = iterate(f.clauses, state...)
        Base.getindex(f::Formula, i::Int) = f.clauses[i]

    end

    """
        get_exact_hamiltonian(formula, n_vars::Int)
    
    Creates an exact Hamiltonian for Max-3-SAT without ancilla qubits.
    This leads to up to 3-body Z-Z-Z interactions (RZZZ terms).
    
    
    # Arguments
    - `formula`: Max3SAT formula from Types
    - `n_vars::Int`: Number of variables
    
    # Returns
    - Vector{ScaledPauli} representing the exact Hamiltonian
    """
    function get_exact_hamiltonian(formula::Types.Formula, n_vars::Int)
        H_sum = PauliSum(n_vars)
        
        # Function to convert a literal to Pauli operator
        function literal_penalty_operator(lit, n)
            sign = lit.neg ? -1.0 : 1.0
            op = PauliSum(n)
            op += 0.5 * Pauli(n)  # Identity term
            z_pauli = Pauli(n; Z=[lit.var])
            op += (0.5 * sign) * z_pauli
            return op
        end
        
        for clause in formula.clauses
            lits = clause.lits
            p1 = literal_penalty_operator(lits[1], n_vars)
            p2 = literal_penalty_operator(lits[2], n_vars)
            p3 = literal_penalty_operator(lits[3], n_vars)
            
            clause_H = p1 * p2 * p3
            H_sum += clause_H
        end
        
        # Return as Vector{ScaledPauli} like ADAPT.jl expects
        hamiltonian = [ScaledPauli(coeff, pauli) for (pauli, coeff) in H_sum.ops]
        return hamiltonian
    end

end