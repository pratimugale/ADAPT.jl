"""
KaMIS Integration Module

This module provides functions to integrate KaMIS's mmwis algorithm
for maximum weight independent set selection in TETRIS ADAPT.
"""

import ..ADAPT
import PauliOperators: ScaledPauliVector

# Helper functions (also defined in TETRIS.jl, but we define them here for independence)
# checked
# TODO: layer use the function from TETRIS.jl
function support(spv::ScaledPauliVector)
    indices = Set{Int64}()
    for sp in spv
        op_indices=findall(x -> x != 'I',string(sp.pauli))
        union!(indices,op_indices)
    end
    return indices
end

# TODO: later use the function from TETRIS.jl
function get_pauli_type(spv::ScaledPauliVector)
    # Get the Pauli string representation (e.g., "XX", "YY", "ZZ", "XY", etc.)
    # For ScaledPauliVector, we combine all Pauli terms
    pauli_strings = String[]
    for sp in spv
        pauli_str = string(sp.pauli)
        # Remove 'I' characters and get only the non-identity Paulis
        non_identity = filter(c -> c != 'I', pauli_str)
        if !isempty(non_identity)
            push!(pauli_strings, non_identity)
        end
    end
    return join(pauli_strings, " + ")  # If multiple terms, join them
end

"""
    get_operator_name(op::ScaledPauliVector)

Generate a unique name for an operator based on its support and Pauli type.
This function is meant for debugging and serves no logical functionality.
Format: "support_pauli" (e.g., "12_XX", "1_X")
"""
function get_operator_name(op::ScaledPauliVector)
    op_support = support(op)
    support_str = join(sort(collect(op_support)), "")
    pauli_type = get_pauli_type(op)
    # Clean up pauli_type to be a valid identifier (remove spaces, special chars)
    pauli_clean = replace(pauli_type, " " => "", "+" => "_", "-" => "m")
    return "$(support_str)_$(pauli_clean)"
end

"""
    build_operator_graph(pool, scores, gradient_threshold)

Build a graph representation of operators where:
- Each operator is a node
- Nodes are connected by edges if their supports overlap
- Each node has a weight equal to its gradient/score

Returns:
- node_to_operator: Dict mapping node index to operator
- node_to_score: Dict mapping node index to score
- adjacency_list: Vector of vectors representing edges
- node_names: Vector of node names (for debugging)
"""
function build_operator_graph(pool, scores, gradient_threshold)
    # Filter operators above threshold
    valid_ops = []
    valid_scores = []
    valid_indices = []
    
    # remove operators with scores below the threshold
    for (idx, (op, score)) in enumerate(zip(pool, scores))
        if score >= gradient_threshold
            push!(valid_ops, op)
            push!(valid_scores, score)
            push!(valid_indices, idx)
        end
    end
    
    n_nodes = length(valid_ops)
    println("Building graph with $n_nodes nodes (operators above threshold)")
    
    # Early return if no operators pass threshold
    if n_nodes == 0
        return Dict(), Dict(), Vector{Int}[], String[], Int[]
    end
    
    # Build adjacency list: nodes are connected if their supports overlap. Different generator type (X_1 X_2) and (Y_1 Y_2) are connected as they have the same support, thus this is opeartor agnostic.

    # adj list is a list of lists, where the i-th list contains the indices of the nodes that are connected to the i-th node.
    # The i-th node is the i-th operator in the valid_ops list.
    adjacency_list = [Int[] for _ in 1:n_nodes]

    # valid_ops[i] is the index of the i-th operator in the original pool. i is the index of the i-th operator in the valid_ops list.
    # node_to_operator is a dictionary that maps the index of the node of the graph to the operator in the original pool.
    node_to_operator = Dict(i => valid_ops[i] for i in 1:n_nodes)

    # valid_scores[i] is the score of the i-th operator in the original pool. i is the index of the i-th operator in the valid_ops list.
    # node_to_score is a dictionary that maps the index of the node of the graph to the score of the operator in the original pool.
    node_to_score = Dict(i => valid_scores[i] for i in 1:n_nodes)

    # node_names is a list of strings, where the i-th string is the name of the i-th operator in the original pool.
    # The name is a unique identifier for the operator, constructed from its support and Pauli type.
    # Note that this is never used in the core logic of KaMIS and Tetris-with-KaMIS, but is used for debugging and visualization.
    node_names = [get_operator_name(valid_ops[i]) for i in 1:n_nodes]
    
    # Compute supports once for all operators in the valid_ops list.
    supports = [support(op) for op in valid_ops]
    
    # Build edges: connect nodes if their supports overlap
    for i in 1:n_nodes
        for j in (i+1):n_nodes
            if !isdisjoint(supports[i], supports[j])
                push!(adjacency_list[i], j)
                push!(adjacency_list[j], i)
            end
        end
    end
    
    # Count edges (each edge is counted twice, once per endpoint)
    total_edges = sum(length(adj) for adj in adjacency_list) ÷ 2
    println("Graph has $total_edges edges")
    
    return node_to_operator, node_to_score, adjacency_list, node_names, valid_indices
end

"""
    write_metis_graph(adjacency_list, weights, output_file)

    adjacency_list: Vector of vectors representing edges
    weights: Vector of weights for each node
    output_file: String path to the output file
Write a graph in METIS format.
- First line: n_nodes n_edges 10
  where 10 means node weights only
- For each node: node_weight neighbor1 neighbor2 ...
  (neighbors are 1-indexed in METIS format)
"""
function write_metis_graph(adjacency_list, weights, output_file)
    n_nodes = length(adjacency_list)
    
    # Clean and count edges: ensure neighbors are sorted, unique, and no self-loops
    # cleaned_adjacency = Vector{Int}[]
    # Since adjacency_list is already clean, just compute total_edges
    total_edges = sum(length(adj) for adj in adjacency_list)
    # for i in 1:n_nodes
    #     # Sort and ensure unique neighbors, remove self-loops
    #     # TODO: do we need this later? It could make this function more generic
    #     # neighbors = sort(unique(filter(n -> n != i, adjacency_list[i])))
    #     neighbors = adjacency_list[i]  # Already clean from build_operator_graph logic
    #     push!(cleaned_adjacency, neighbors)
    #     total_edges += length(neighbors)
    # end
    n_edges = total_edges ÷ 2
    
    # Verify edge count matches
    if total_edges % 2 != 0
        @warn "Odd number of edge entries ($total_edges), graph may have issues"
    end
    
    open(output_file, "w") do f
        # Write header: n_nodes n_edges ew
        # KaMIS uses combined format code:
        # ew=0: no weights, ew=1: edge weights, ew=10: node weights, ew=11: both
        # We use ew=10 for node weights only (vertex weights)
        # Format must be exactly: "n_nodes n_edges 10\n" with no extra spaces
        header_line = "$n_nodes $n_edges 10"
        println(f, header_line)
        
        # Write each node: weight neighbor1 neighbor2 ...
        # METIS uses 1-indexed vertices, which matches our indexing
        # KaMIS expects integer weights - weights should already be Int at this point
        for i in 1:n_nodes
            weight_int = weights[i]  # Weights are already integers
            neighbors = adjacency_list[i]  # Already clean from build_operator_graph logic
            if isempty(neighbors)
                println(f, "$weight_int")
            else
                neighbor_str = join(neighbors, " ")
                println(f, "$weight_int $neighbor_str")
            end
        end
    end
    
    # Verify the file was written correctly by reading the first few lines
    open(output_file, "r") do f
        first_line = readline(f)
        println("METIS header written: '$first_line' (expected: '$n_nodes $n_edges 10')")
        # Read and print first 3 data lines to verify format
        for i in 1:min(3, n_nodes)
            if !eof(f)
                data_line = readline(f)
                println("  Line $i: '$data_line'")
            end
        end
    end
    
    println("Wrote METIS graph to $output_file (nodes: $n_nodes, edges: $n_edges)")
end

"""
    run_kamis_mmwis(graph_file::String; 
                    kamis_path::String="",
                    output_file::String="",
                    seed::Int=0,
                    time_limit::Float64=0.0,
                    config::String="mmwis")

Run KaMIS mmwis algorithm on a graph file.

Returns the path to the output file containing the independent set.
"""
function run_kamis_mmwis(graph_file::String; 
                         kamis_path::String="",
                         output_file::String="",
                         seed::Int=0,
                         time_limit::Float64=0.0,
                         config::String="mmwis")
    
    # Default KaMIS path (can be overridden)
    if isempty(kamis_path)
        # Try common locations
        possible_paths = [
            joinpath(pwd(), "KaMIS", "deploy", "mmwis"),  # Most likely location after compilation
            joinpath(homedir(), "KaMIS", "deploy", "mmwis"),
            joinpath(homedir(), "KaMIS", "mmwis"),
            joinpath(pwd(), "KaMIS", "mmwis"),
            "mmwis",  # If in PATH
        ]
        
        kamis_path = ""
        for path in possible_paths
            if isfile(path) || (isfile("$path.exe") && Sys.iswindows())
                kamis_path = path
                break
            end
        end
        
        if isempty(kamis_path)
            error("KaMIS mmwis executable not found. Please specify kamis_path or ensure mmwis is in PATH")
        end
    end
    
    # Generate output file name if not provided
    if isempty(output_file)
        output_file = replace(graph_file, ".graph" => "_solution.txt", ".metis" => "_solution.txt")
        if output_file == graph_file
            output_file = "$(graph_file)_solution.txt"
        end
    end
    
    # Build command - use proper command construction
    cmd = `$kamis_path $graph_file --output=$output_file --config=$config`
    
    if seed > 0
        cmd = `$cmd --seed=$seed`
    end
    
    if time_limit > 0
        cmd = `$cmd --time_limit=$time_limit`
    end
    
    println("Running KaMIS mmwis: $cmd")
    
    # Run command - in Julia, run() throws an error if the process fails
    # So we just need to catch and re-throw with better error message
    try
        run(cmd)
    catch e
        error("KaMIS mmwis failed: $e")
    end
    
    return output_file
end

"""
    parse_kamis_solution(solution_file::String; n_nodes::Union{Int, Nothing}=nothing)

Parse the solution file from KaMIS mmwis.
The solution file contains the independent set as vertex indices (1-indexed).

Returns a vector of node indices (1-indexed) that form the independent set.
"""
function parse_kamis_solution(solution_file::String; n_nodes::Union{Int, Nothing}=nothing)
    if !isfile(solution_file)
        error("Solution file not found: $solution_file")
    end
    
    # KaMIS writeIndependentSet writes one value per line:
    # 1 if the node (1-indexed by line number) is in the independent set
    # 0 otherwise
    selected_nodes = Int[]
    node_idx = 1  # KaMIS uses 1-indexed nodes
    open(solution_file, "r") do f
        for line in eachline(f)
            line = strip(line)
            if isempty(line)
                continue
            end
            
            # Parse the value (should be 0 or 1)
            value = tryparse(Int, line)
            if value !== nothing && value == 1
                push!(selected_nodes, node_idx)
            end
            node_idx += 1
        end
    end
    
    # After reading all lines, verify we got the expected number
    if n_nodes !== nothing && node_idx - 1 != n_nodes
        @warn "Solution file has $(node_idx - 1) lines, expected $n_nodes nodes"
    end
    
    return selected_nodes
end

"""
    select_operators_with_kamis(pool, scores, gradient_threshold;
                                kamis_path::String="",
                                temp_dir::String="",
                                seed::Int=0)

Main function to select operators using KaMIS mmwis algorithm.

Returns:
- selected_operator_indices: Indices into the original pool
- selected_operators: The actual operators
- selected_scores: The scores of selected operators
"""
function select_operators_with_kamis(pool, scores, gradient_threshold;
                                      kamis_path::String="",
                                      temp_dir::String="",
                                      seed::Int=0)
    
    # Build graph
    node_to_operator, node_to_score, adjacency_list, node_names, valid_indices = 
        build_operator_graph(pool, scores, gradient_threshold)
    
    if isempty(node_to_operator)
        println("No operators above threshold, returning empty selection")
        return Int[], [], Float64[]
    end
    
    # Prepare weights (scores)
    n_nodes = length(node_to_operator)
    weights = [node_to_score[i] for i in 1:n_nodes]
    
    # First multiply by 10^14 to convert to integer with high precision
    SCALE_FACTOR = 1.0e14
    MAX_SAFE_SUM = 2000000000.0  # 2e9 (safe limit, well below Int64 max)
    
    # Multiply by scale factor
    scaled_weights = [w * SCALE_FACTOR for w in weights]
    
    # Calculate total sum
    total_sum = sum(scaled_weights)
    
    # Scale down if sum exceeds safe limit
    if total_sum > MAX_SAFE_SUM
        scale_down_factor = MAX_SAFE_SUM / total_sum
        scaled_weights = [w * scale_down_factor for w in scaled_weights]
        println("Warning: Sum of scaled weights ($(total_sum)) exceeded safe limit. Scaled down by factor $(scale_down_factor) to preserve relative ratios.")
    end
    
    # Convert to integers
    weight_ints = [round(Int, w) for w in scaled_weights]
    # Ensure non-zero weights stay non-zero
    for i in 1:length(weight_ints)
        if weight_ints[i] == 0 && weights[i] > 0
            weight_ints[i] = 1
        end
    end
    
    # Create temporary directory if needed
    if isempty(temp_dir)
        temp_dir = mktempdir()
    end
    
    # Write graph to METIS format
    graph_file = joinpath(temp_dir, "operators_$(rand(UInt64)).metis")
    write_metis_graph(adjacency_list, weight_ints, graph_file)
    
    # Run KaMIS mmwis
    solution_file = run_kamis_mmwis(graph_file; kamis_path=kamis_path, seed=seed)
    
    # Parse solution
    selected_node_indices = parse_kamis_solution(solution_file; n_nodes=n_nodes)
    
    println("KaMIS selected $(length(selected_node_indices)) operators")
    
    # Map back to original pool indices
    selected_pool_indices = [valid_indices[node_idx] for node_idx in selected_node_indices]
    selected_operators = [node_to_operator[node_idx] for node_idx in selected_node_indices]
    selected_scores = [node_to_score[node_idx] for node_idx in selected_node_indices]
    
    # Clean up temporary files (optional)
    # rm(graph_file, force=true)
    # rm(solution_file, force=true)
    
    return selected_pool_indices, selected_operators, selected_scores
end

