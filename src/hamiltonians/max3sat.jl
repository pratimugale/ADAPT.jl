module Max3SAT
    
    import PauliOperators: Pauli, PauliSum, ScaledPauli

    export Types

    """
    Submodule containing the structural definitions for SAT formulas.
    """
    module Types

        """
        Literal(var, neg): Represents a boolean variable.
        - `var`: 1-based index of the qubit.
        - `neg`: If true, represents (NOT x); if false, represents (x).
        """
        struct Literal
            var::Int
            neg::Bool
        end

        """
        Clause(lits): A disjunction (OR) of exactly 3 literals.
        Stored as an NTuple for memory efficiency.
        """
        struct Clause
            lits::NTuple{3, Literal}
        end

        """
        Formula(clauses): A conjunction (AND) of multiple clauses.
        Acts as the container for the Max-3-SAT problem instance.
        """
        struct Formula
            clauses::Vector{Clause}
        end

        # Treat Formula like a collection of Clause
        Base.length(f::Formula) = length(f.clauses)
        Base.iterate(f::Formula, state...) = iterate(f.clauses, state...)
        Base.getindex(f::Formula, i::Int) = f.clauses[i]
 
        """
            get_formula_from_list(formula_list::Vector)
        Create a Formula struct from a list of lists of integers (DIMACS-style literals).
        Positive integer x -> x-th variable.
        Negative integer -x -> NOT x-th variable.
        """
        function get_formula_from_list(formula_list::Vector)
            clauses = Clause[]
            for clause_data in formula_list
                lits = Literal[]
                for lit in clause_data
                    var = abs(lit)
                    neg = lit < 0
                    push!(lits, Literal(var, neg))
                end
                if length(lits) != 3
                    error("Max-3-SAT requires exactly 3 literals per clause. Found $(length(lits)).")
                end
                push!(clauses, Clause((lits[1], lits[2], lits[3])))
            end
            return Formula(clauses)
        end

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

    # Approximate Hamiltonian implementation next.
    # We implement the FullApprox implementation from https://arxiv.org/pdf/2409.15891

    # QUBO matrices for each clause type
    const TYPE_A_QUBO = [-1  1  1;
                          0 -1  1;
                          0  0 -1]

    const TYPE_B_QUBO = [ 0  1 -1;
                          0  0 -1;
                          0  0  1]

    const TYPE_C_QUBO = [ 1 -1 -1;
                          0  0  1;
                          0  0  0]

    # Note that the TYPE_D_QUBO is the same as the TYPE_A_QUBO according to the paper.
    const TYPE_D_QUBO = [-1  1  1;
                          0 -1  1;
                          0  0 -1]

    """ 
        convert_binary_literal_to_spin_operator(lit, n)

    Converts a non-negated binary variable to an Ising variable.
    x_n -> (1+Z_n)/2
    """
    function convert_binary_literal_to_spin_operator(lit, n)
        # we don't need to check the sign as the approx QUBO matrix handles the sign
        # just return 1+Z_lit.var/2
        op = PauliSum(n)
        op += 0.5 * Pauli(n)  # Identity term
        z_pauli = Pauli(n; Z=[lit.var])
        op += 0.5 * z_pauli
        return op
    end

    """
        qubo_to_pauli_terms(qubo_matrix, lits, n_vars)

    Converts a QUBO matrix to Pauli terms for Ising Hamiltonian.
    The QUBO polynomial is: Σᵢⱼ Qᵢⱼ xᵢ xⱼ
    This becomes: Σᵢⱼ Qᵢⱼ (1+Zᵢ)(1+Zⱼ)/4
    """
    function qubo_to_pauli_terms(qubo_matrix, lits, n_vars)
        pauli_sum = PauliSum(n_vars)

        for i in 1:3
            for j in 1:3
                coeff = qubo_matrix[i, j]
                if coeff != 0
                    variable_1 = lits[i]
                    variable_2 = lits[j]
                    if i == j
                        # this would be a linear term in the Ising Hamiltonian
                        pauli_sum += coeff * convert_binary_literal_to_spin_operator(variable_1, n_vars)
                    else
                        pauli_sum += coeff * (convert_binary_literal_to_spin_operator(variable_1, n_vars) * convert_binary_literal_to_spin_operator(variable_2, n_vars))
                    end
                end
            end
        end

        return pauli_sum
    end

    """
        process_type_a_clause(clause, n_vars)

    Processes Type A clauses (0 negations - all 3 literals positive).
    """
    function process_type_a_clause(clause, n_vars)
        clause_H = PauliSum(n_vars)
        lits = clause.lits  # Already in correct order (all positive)
        pauli_terms = qubo_to_pauli_terms(TYPE_A_QUBO, lits, n_vars)
        clause_H += pauli_terms
        return clause_H
    end

    """
        process_type_b_clause(clause, n_vars)

    Processes Type B clauses (1 negation).
    Rearranges literals so positive literal comes first, then negative literals.
    """
    function process_type_b_clause(clause, n_vars)

        clause_H = PauliSum(n_vars)

        # Separate positive and negative literals
        positive_lits = [lit for lit in clause.lits if !lit.neg]
        negative_lits = [lit for lit in clause.lits if lit.neg]

        # Rearrange: positive first, then negative
        rearranged_lits = vcat(positive_lits, negative_lits)

        pauli_terms = qubo_to_pauli_terms(TYPE_B_QUBO, rearranged_lits, n_vars)
        clause_H += pauli_terms
        return clause_H
    end

    """
        process_type_c_clause(clause, n_vars)

    Processes Type C clauses (2 negations).
    Rearranges literals so positive literal comes first, then negative literals.
    """
    function process_type_c_clause(clause, n_vars)
        clause_H = PauliSum(n_vars)
        # Separate positive and negative literals
        positive_lits = [lit for lit in clause.lits if !lit.neg]
        negative_lits = [lit for lit in clause.lits if lit.neg]

        # Rearrange: positive first, then negative
        rearranged_lits = vcat(positive_lits, negative_lits)

        pauli_terms = qubo_to_pauli_terms(TYPE_C_QUBO, rearranged_lits, n_vars)
        clause_H += pauli_terms
        return clause_H
    end

    """
        process_type_d_clause(clause, n_vars)

    Processes Type D clauses (3 negations - all 3 literals negative).
    """
    function process_type_d_clause(clause, n_vars)
        clause_H = PauliSum(n_vars)
        lits = clause.lits  # Already in correct order (all negative)
        pauli_terms = qubo_to_pauli_terms(TYPE_D_QUBO, lits, n_vars)
        clause_H += pauli_terms
        return clause_H
    end

    """
        get_approximate_hamiltonian(formula, n_vars::Int)

    Creates an approximate Hamiltonian for Max-3-SAT using clause categorization.
    Each clause is categorized into 4 types based on the number of negations:
    - Type A: 0 negations (all 3 literals positive)
    - Type B: 1 negation
    - Type C: 2 negations  
    - Type D: 3 negations (all 3 literals negative)

    # Arguments
    - `formula`: Max3SAT formula from ADAPT.jl
    - `n_vars::Int`: Number of variables

    # Returns
    - Vector{ScaledPauli} representing the approximate Hamiltonian terms
    """
    function get_approximate_hamiltonian(formula::Types.Formula, n_vars::Int)
        total_H = PauliSum(n_vars)

        # Count clause types for analysis
        type_counts = Dict("A" => 0, "B" => 0, "C" => 0, "D" => 0)

        for clause in formula.clauses
            lits = clause.lits

            # Count negations in this clause
            negations = sum(lit.neg for lit in lits)

            # Categorize clause type
            if negations == 0
                clause_type = "A"
            elseif negations == 1
                clause_type = "B"
            elseif negations == 2
                clause_type = "C"
            else  # negations == 3
                clause_type = "D"
            end

            type_counts[clause_type] += 1

            # Process each clause type with appropriate Hamiltonian terms
            if clause_type == "A"
                clause_H = process_type_a_clause(clause, n_vars)
                total_H += clause_H
            elseif clause_type == "B"
                clause_H = process_type_b_clause(clause, n_vars)
                total_H += clause_H
            elseif clause_type == "C"
                clause_H = process_type_c_clause(clause, n_vars)
                total_H += clause_H
            else  # clause_type == "D"
                clause_H = process_type_d_clause(clause, n_vars)
                total_H += clause_H
            end

        end

        # # Print clause type statistics
        # println("Clause type distribution:")
        # for (type, count) in type_counts
        #     println("  Type $type: $count clauses")
        # end

        # println("total_H: $(total_H)")
        # Convert PauliSum to Vector{ScaledPauli{n_vars}}
        return [ScaledPauli{n_vars}(coeff, pauli) for (pauli, coeff) in total_H.ops]
    end

end