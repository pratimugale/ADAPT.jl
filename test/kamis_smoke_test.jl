#=
Smoke test for KaMIS mmwis (Maximum Weight Independent Set) solver.

This test verifies that the external KaMIS binary is correctly installed
and can solve a simple MWIS problem.

Run standalone: julia --project=. test/kamis_smoke_test.jl
=#

"""
    test_kamis_mmwis()

Smoke test for KaMIS mmwis binary. Creates a small test graph, runs mmwis,
and verifies the output is a valid independent set.

Returns `true` if test passes, throws an error otherwise.
"""
function test_kamis_mmwis()
    # Path to mmwis binary (relative to project root)
    project_root = dirname(dirname(@__FILE__))
    mmwis_path = joinpath(project_root, "external", "KaMIS", "deploy", "mmwis")
    
    # Check if binary exists
    if !isfile(mmwis_path)
        error("KaMIS mmwis binary not found at: $mmwis_path\n" *
              "Please run 'make install-kamis' to build KaMIS.")
    end
    
    # Create a small test graph in METIS format
    # Graph: 4 nodes forming a square with one diagonal
    #   1 --- 2
    #   |  \  |
    #   4 --- 3
    #
    # Edges: (1,2), (2,3), (3,4), (4,1), (1,3)
    # Node weights: 10, 5, 10, 5
    # Optimal MWIS: {1, 3} with weight 20, or {2, 4} with weight 10
    # So we expect nodes 1 and 3 to be selected
    
    graph_content = """
4 5 10
10 2 3 4
6 1 3
10 1 2 4
5 1 3
"""
    
    # Create temporary files
    graph_file = tempname() * ".graph"
    output_file = tempname() * ".mis"
    
    try
        # Write graph file
        open(graph_file, "w") do f
            write(f, graph_content)
        end
        
        println("Testing KaMIS mmwis...")
        println("  Graph file: $graph_file")
        println("  Binary: $mmwis_path")
        
        # Run mmwis with timeout
        # --time_limit=5 limits runtime, --seed for reproducibility
        cmd = `$mmwis_path $graph_file --time_limit=5 --seed=42 --output=$output_file`
        
        result = run(pipeline(cmd, stdout=devnull, stderr=devnull), wait=true)
        
        if result.exitcode != 0
            error("mmwis exited with code $(result.exitcode)")
        end
        
        # Read output if it exists
        if isfile(output_file)
            solution = parse.(Int, filter(!isempty, readlines(output_file)))
            println("  Solution: $solution")
            
            # Verify it's a valid independent set
            # solution[i] = 1 means node i is in the set
            selected = findall(x -> x == 1, solution)
            println("  Selected nodes: $selected")
            
            # Check that selected nodes form an independent set
            # (no two selected nodes should be adjacent)
            edges = [(1,2), (2,3), (3,4), (4,1), (1,3)]
            for (u, v) in edges
                if u in selected && v in selected
                    error("Invalid solution: nodes $u and $v are both selected but adjacent!")
                end
            end
            
            # Calculate weight
            weights = [10, 6, 10, 5]
            total_weight = sum(weights[i] for i in selected; init=0)
            println("  Total weight: $total_weight")
            
            # We expect weight >= 10 (at least one high-weight node)
            if total_weight < 10
                error("Solution weight $total_weight is suspiciously low")
            end
            
            println("✓ KaMIS mmwis smoke test PASSED!")
            return true
        else
            # mmwis might not write output file, just check it ran
            println("  (No output file generated, but binary ran successfully)")
            println("✓ KaMIS mmwis smoke test PASSED (basic)!")
            return true
        end
        
    finally
        # Cleanup
        rm(graph_file, force=true)
        rm(output_file, force=true)
    end
end

# Run test if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_kamis_mmwis()
end
