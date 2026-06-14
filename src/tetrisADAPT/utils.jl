# Contains utility functions for TETRIS (needed by both KaMIS.jl and TETRIS.jl)

import PauliOperators: ScaledPauliVector

"""
    Get the support of a ScaledPauliVector
    Here, we define support as the indices of the Pauli operators that are not identity ('I').
"""
function support(spv::ScaledPauliVector)
    indices = Set{Int64}()
    for sp in spv
        op_indices=findall(x -> x != 'I',string(sp.pauli))
        union!(indices,op_indices)
    end
    return indices
end

"""
    Get the Pauli string representation of a ScaledPauliVector (e.g., "XX", "YY", "ZZ", "XY", etc.)
    This is needed for KaMIS, to make the names of the operators easy to read.
    A ScaledPauliVector is technically a sum of Pauli strings, but we combine them into a single string using a "+".
    For example, if the ScaledPauliVector is [XX, YY, ZZ], we return "XX + YY + ZZ".
    If the spv is an identity operator, we return "". This is because in practice, there wouldn't be 
    any operators with support 0, so we can use this to represent the identity operator.
"""
function get_pauli_type(spv::ScaledPauliVector)
    # For ScaledPauliVector, we combine all Pauli terms
    pauli_strings = String[]
    for sp in spv
        pauli_str = string(sp.pauli) # See https://github.com/nmayhall-vt/PauliOperators.jl/blob/0d94113e7e040fd3dd447d37ccf35be03ce5400e/src/type_Pauli.jl#L159C1-L180C4 for understanding the string conversion
        # TODO: do we care about the coefficients? y = iY according to the docs of the PauliOperators.jl
        # However, we just want to know the support of the operator to create the incompatibility graph, so we can ignore the coefficients
        # The support function above (original greedy) method also seems to ignore the coefficients

        # Remove 'I' characters and get only the non-identity Paulis
        non_identity = filter(c -> c != 'I', pauli_str)
        if !isempty(non_identity)
            push!(pauli_strings, non_identity)
        end
    end
    return join(pauli_strings, " + ")  # If multiple terms, join them with a "+"
end

"""
    get_operator_name(op::ScaledPauliVector)

Generate a unique name for an operator based on its support and Pauli type.
This function is meant for debugging and serves no logical functionality.
Format: "support_pauli" (e.g., "12_XX", "1_X")
TODO: this function can be made better later
"""
function get_operator_name(op::ScaledPauliVector)
    op_support = support(op)
    support_str = join(sort(collect(op_support)), "")
    pauli_type = get_pauli_type(op)
    # Clean up pauli_type to be a valid identifier (remove spaces, special chars)
    # TODO: The cleaned name can be formatted better. For now its fine as it is used only for debugging.
    pauli_clean = replace(pauli_type, " " => "", "+" => "_", "-" => "m")
    return "$(support_str)_$(pauli_clean)"
end


