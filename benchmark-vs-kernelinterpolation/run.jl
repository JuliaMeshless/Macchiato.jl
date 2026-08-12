# Entry point: verify setting parity, then tune and run the suite, save results.json, and
# print a comparison table.
# Run: julia --project=benchmark-vs-kernelinterpolation benchmark-vs-kernelinterpolation/run.jl

include("common.jl")
include("benchmarks.jl")
include("verify.jl")

using Printf
using BenchmarkTools

# Display order (BenchmarkGroup itself is unordered)
const BENCH_ORDER = [
    "interpolation 1D" => ["KernelInterpolation", "Macchiato stack"],
    "RBF-FD operator build 2D" => [
        "KernelInterpolation (Lagrange basis)",
        "KernelInterpolation (standard basis)",
        "Macchiato stack",
    ],
    "RBF-FD Poisson 2D" => [
        "KernelInterpolation (Lagrange basis)",
        "KernelInterpolation (standard basis)",
        "Macchiato stack",
    ],
    "RBF-FD Poisson 2D end-to-end" => ["KernelInterpolation", "Macchiato stack"],
    "RBF-FD Advection 2D rhs!" => ["KernelInterpolation", "Macchiato stack"],
]

prettytime(ns::Float64) =
    ns < 1.0e3 ? @sprintf("%.1f ns", ns) :
    ns < 1.0e6 ? @sprintf("%.2f μs", ns / 1.0e3) :
    ns < 1.0e9 ? @sprintf("%.2f ms", ns / 1.0e6) : @sprintf("%.2f s", ns / 1.0e9)

prettymem(bytes::Float64) =
    bytes < 1024 ? @sprintf("%.0f B", bytes) :
    bytes < 1024^2 ? @sprintf("%.1f KiB", bytes / 1024) : @sprintf("%.2f MiB", bytes / 1024^2)

function print_results(results)
    println("\n", "="^100)
    @printf("%-44s %14s %14s %12s %10s\n", "benchmark / implementation", "median", "mean", "memory", "allocs")
    println("="^100)
    for (group_name, entries) in BENCH_ORDER
        group = results[group_name]
        println(group_name)
        ki_best = minimum(time(median(group[e])) for e in entries if startswith(e, "KernelInterpolation"))
        for entry in entries
            t = group[entry]
            med = time(median(t))
            @printf(
                "  %-42s %14s %14s %12s %10d\n",
                entry, prettytime(med), prettytime(time(mean(t))),
                prettymem(Float64(memory(minimum(t)))), allocs(minimum(t)),
            )
        end
        ours_med = time(median(group["Macchiato stack"]))
        @printf(
            "  → ours is %.2f× %s than the fastest KI variant\n\n",
            ours_med < ki_best ? ki_best / ours_med : ours_med / ki_best,
            ours_med < ki_best ? "faster" : "slower"
        )
    end
    return
end

verify()

println("Tuning benchmark suite…")
tune!(SUITE)
println("Running benchmark suite…")
results = run(SUITE; verbose = true)

outfile = joinpath(@__DIR__, get(ENV, "BENCH_RESULTS_FILE", "results.json"))
BenchmarkTools.save(outfile, results)
println("\nSaved results to ", outfile)

print_results(results)
