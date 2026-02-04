
@testset "QAOA Nondiagonal Double Pool Completeness" begin
    # We need to access the pool creation function.
    # Assuming it is exported or we can reach it via ADAPT.ADAPT_QAOA.QAOApools
    QAOApools = ADAPT.ADAPT_QAOA.QAOApools
    
    n = 3
    pool = QAOApools.qaoa_nondiagonal_double_pool(n)
    
    # 1. Check size
    # Expected: 2N (single X, Y) + 1 (Mixer) + 8 * N(N-1) / 2 (Doubles)
    # For N=3: 6 + 1 + 24 = 31
    expected_size = 2*n + 1 + 4*n*(n-1)
    
    @test length(pool) == expected_size
    
    # 2. Check for diagonal operators (Strictly Z or I)
    for (i, op_vec) in enumerate(pool)
        for scaled_pauli in op_vec
            p = scaled_pauli.pauli
            # if p.x != 0, it is not I or Z
            # The operator is X if p.x != 0 and p.z == 0
            # The operator is Y if p.x != 0 and p.z != 0
            # The operator is Z if p.x == 0 and p.z != 0
            @test p.x != 0
        end
    end
    
    # 3. Check for presence of specific expected types
    count_single_X = 0
    count_single_Y = 0
    count_mixer = 0
    count_double = 0
    
    for op_vec in pool
        if length(op_vec) > 1
            # Must be the mixer
            # Verify it is Sum X_k
            @test length(op_vec) == n
            count_mixer += 1
        else
            # Single term
            sp = op_vec[1]
            p = sp.pauli
            
            weight = count_ones(p.x | p.z) # Total non-identity sites
            
            if weight == 1
                if p.x != 0 && p.z == 0 # Pure X
                    count_single_X += 1
                elseif p.x != 0 && p.z != 0 # Y (X and Z both set)
                    count_single_Y += 1
                end
            elseif weight == 2
                count_double += 1
            end
        end
    end
    
    @test count_single_X == n
    @test count_single_Y == n
    @test count_mixer == 1
    @test count_double == 4*n*(n-1)

end
