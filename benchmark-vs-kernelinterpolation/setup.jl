# One-time environment setup for the KernelInterpolation.jl comparison benchmarks.
# Run: julia benchmark-vs-kernelinterpolation/setup.jl
using Pkg

Pkg.activate(@__DIR__)

# Our stack: Macchiato is unregistered — develop the repo one level up. WhatsThePoint mirrors
# Macchiato's [sources] entry (it tracks main, ahead of the registered release).
Pkg.develop(path = dirname(@__DIR__))
Pkg.add(url = "https://github.com/JuliaMeshless/WhatsThePoint.jl", rev = "main")

# KernelInterpolation's RBF-FD machinery is unreleased — pin the exact commit the reference
# benchmark suite (benchmark/benchmarks.jl) was taken from.
Pkg.add(
    url = "https://github.com/JoshuaLampert/KernelInterpolation.jl",
    rev = "e36034c3f1e0bf1e9393c8e63740c058629c6aca",
)

# RadialBasisFunctions is dev'ed rather than added: registered 0.7.0 has a bug where the
# fused PHS ∇² functors hardcode the 3D Laplacian constants (wrong in 2D), which breaks
# parity for every Laplacian-based benchmark. The local copy must have the
# fix/phs-laplacian-dimension branch (commit 1e26422f) checked out.
Pkg.develop(path = joinpath(homedir(), "dev", "RadialBasisFunctions"))

Pkg.add([
    "BenchmarkTools",
    "LinearSolve",
    "OrdinaryDiffEqRosenbrock",
    "OrdinaryDiffEqNonlinearSolve",
    "SciMLBase",
    "StaticArrays",
    "Unitful",
    "SparseArrays",
    "LinearAlgebra",
    "Random",
    "Printf",
])

Pkg.precompile()
Pkg.status()
