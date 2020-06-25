using iccg
using Test

@testset "iccg.jl" begin
    A = [1 0 0; 0 2 0; 0 0 1]
    b = [4; 5; 6]
    x0 = zeros(Float64, size(b))
    x = ICCG(A, x0, b, 1e+5, 1e-5)
    @test x == [4.00000057924488; 2.499997761223719; 6.000000868867321]

end
