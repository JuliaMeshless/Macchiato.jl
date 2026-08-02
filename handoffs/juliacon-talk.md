# JuliaCon talk — slide sketches

**Talk:** the JuliaMeshless ecosystem (WhatsThePoint.jl · RadialBasisFunctions.jl · Macchiato.jl)
**Slot:** ~30 min · **Deck:** [Google Slides](https://docs.google.com/presentation/d/1byTQvj9ef4w2Ad8wbsGKNE7nUzX5nQSwtRerMupRvSo)
**Built by:** Kyle Beggs and Davide Miotti · **Presented by:** Davide, solo

Working doc for building the deck. Sketches only — the slides get made by hand from
these. Covers the **Overview block** so far; later blocks get appended here.

> On the deck: the "Kyle" / "Davide" labels on individual slides mark who *builds*
> that slide, not who delivers it.

---

## Framing

Every slide gets read against this filter before it goes in.

**Say**
- high impact, ambitious
- presenting an ecosystem, not a single package — more powerful
- casting a wider net for contributors — experts in solvers, geometry, etc.
- novelty; fills a gap in the Julia solvers landscape
- not a port — Julia from the ground up
- not a replacement for other methods, but filling a gap. "the correct tool for the
  job" — sometimes it's better to use FE or FV

**Don't say**
- that it's more accurate or "better" than other methods/packages
- that this ecosystem is only for researchers or only for engineers — it's both
- that future tasks are too complicated and that's why we haven't done them
- that deterministic/traditional solvers are better than AI — they should coexist

---

## The story

The deck already implies a clean five-act arc.

| Act | Slides | Time | Job |
|---|---|---|---|
| **Premise** | 1–5 | 2 min | geometry is a wall; get rid of the mesh |
| **Promise** | 6–8 | 3 min | show it in code, before explaining any machinery |
| **Machinery** | 9–10 | 6 min | earn it — points, operators, convergence |
| **Proof** | 11–12 | 7 min | two real problems, end to end |
| **Ask** | 13 | 2 min | what's next, what's broken, come help |

The through-line is one sentence, and it's already sitting on deck slide 5 in two
halves:

> **Get rid of the mesh, and the code you write starts looking like the problem you
> want to solve.**

Every slide either sets that up, demonstrates it, or backs it with evidence.

**The objection the talk has to survive:** "no meshing" sounds too good, and the
reflex in the room will be *"so it must be less accurate."* Acts 3 and 4 exist to
defuse that. This is why saying plainly that we are **not** claiming better accuracy
is a strength rather than a hedge — it answers the objection before anyone raises it.

Two structural notes. The Overview is the only block asked to carry seven topics in
two minutes, so **"goals" moves to slide 13**, where "future directions" already
lives. And the ecosystem table staying at slide 8 — *after* the API slides — is
right: it lands better as "and here's why that was possible" than as a table of
contents up front.

---

## Overview block — replaces deck slides 4–5

Four slides, ~2 minutes total. Deck slides 4 and 5 are planning notes, not slides;
these replace them.

### O1 — Get rid of the mesh · ~30s

**On slide**

> Everything follows from one design decision: **get rid of the mesh.**
>
> A point cloud and a set of RBF-FD operators is all the backend you need to
> discretize most PDEs.

Footer strip, one line: **JuliaMeshless** — WhatsThePoint.jl · RadialBasisFunctions.jl ·
Macchiato.jl · MIT · all registered in General.

**Visual** — the STL-with-point-cloud hero image. This frame is the whole talk in one
picture; reuse it later.

**Speaker** — Open with the design decision, not with us. "Two people, three
packages," then move. Do **not** tour the packages here: slide 8 is the ecosystem
table, and WhatsThePoint and RadialBasisFunctions each get 3 minutes later.

**Serves** — who we are / what we do · ecosystem not a package.

---

### O2 — Why meshless · ~40s

**On slide** — two columns, same geometry pictured twice.

| Mesh | Points |
|---|---|
| negative volumes, skewed cells | spacing |
| explicit connectivity to repair | k nearest neighbors |
| remesh when the geometry moves | move the points |
| refine = rebuild | refine = **add points** |

Footer line: *Especially when the geometry never came from CAD — a scan, a segmented
medical image.*

**Visual** — a genuinely bad mesh beside the same geometry as a point cloud. A real
screenshot of a meshing failure beats any diagram; use one from our own cardiac work
if we have it.

**Speaker** — The longest beat in the block, and worth it. This is the emotional core;
everything after only lands if the audience feels this one. Consider a show of hands:
*"who here has lost a week to meshing?"*

**Serves** — why/what is meshless · ease of geometry handling.

---

### O3 — Why build a new deterministic solver? · ~30s

**On slide**

- Julia already has FEM, FVM, DG, spectral, structured FD, PINNs.
  **Every one of them needs a mesh or a grid.**
- Meshless RBF-FD is the empty row. That's the gap — not a claim to beat anyone.
- And a point cloud **is** a graph: a substrate for GNNs and differentiable physics,
  not a rival to them. RadialBasisFunctions.jl ships Enzyme and Mooncake rules and a
  LuxCore `RBFLayer` today.

**Visual** — one point cloud with the neighbor edges drawn, labeled twice: "stencil"
and "graph." Same picture, both readings.

**Speaker** — This is where the ML question gets answered, and it should sound like an
invitation to the ML people in the room, not a defense. If KernelInterpolation.jl
comes up in Q&A, be warm about it — global dense collocation gives spectral accuracy
at modest N; our local RBF-FD gives a sparse system that reaches 10⁵–10⁶ nodes on real
3D geometry. Different tools, different problems.

**Serves** — why bother · fills a gap · AI coexistence · wider net for contributors.

---

### O4 — What we are / what we are not · ~25s

**On slide**

| We are | We are not |
|---|---|
| a general-purpose meshless PDE framework | a replacement for FEM or FVM |
| Julia from the ground up — not a port | a claim to be more accurate |
| composable — use any layer on its own | a black box |
| for researchers **and** engineers | finished |

Closing line: **Geometry stops being the hard part. Bring a shape — from CAD, a scan,
a medical image — and go straight to physics.**

**Speaker** — Say "finished" plainly and without apologising; it's the setup for slide
13's call for help. Frame every gap as open, well-defined work someone could pick up,
never as "too complicated, that's why we haven't done it." Then hand straight into the
API section.

**Serves** — what we are / are not · vision · right-tool-for-the-job · both audiences.

---

## Verified facts

Checked against the repos on 2026-08-02, so nothing here needs re-deriving and no
slide claim is unsourced.

**Packages** — all three registered in General.

| Package | Version | Notes |
|---|---|---|
| RadialBasisFunctions.jl | 0.6.0 | 10 releases, Zenodo DOI, AirspeedVelocity benchmark CI, published convergence + work-precision study in `docs/src/assets/convergence/` |
| WhatsThePoint.jl | 0.2.0 | 4 node-placement algorithms with citations; PCA normals + MST orientation (Hoppe 1992); octree SDF with pseudonormals (Bærentzen–Aanæs 2005); RBF implicit surfaces (Carr 2001) |
| Macchiato.jl | 0.1.0 | physics layer |

**People** — 78 commits Kyle, 43 Davide. The honest basis for "two people."

**Physics implemented** — `SolidEnergy` (steady + transient conduction),
`LinearElasticity` (2D plane stress), `IncompressibleNavierStokes` (in development).

**Validation available** — `examples/cantilever_beam_2d_ferrite.jl` cross-checks
against Ferrite.jl; the elasticity tests include a patch test to <1e-10 and a
3-resolution MMS convergence test; `2d_Laplacian_MoMS.jl` is an MMS Poisson with mixed
Dirichlet/Neumann/Robin; `examples/niederer_benchmark/` runs the Niederer N-version
cardiac electrophysiology benchmark with a ten Tusscher–Panfilov 2006 ionic model.

**Positioning vs. KernelInterpolation.jl** — already written up in
`comparison_kernelinterpolation.md`. Global dense collocation vs. our local sparse
RBF-FD. Being generous here reads as confidence and is the best
contributor-recruiting move in the block.

---

## Flags

1. **Test the live demo early.** RadialBasisFunctions.jl v0.6.0 removed the positional
   `eval_points` operator constructors, but Macchiato still calls the old form in
   `src/upwinding.jl:34` and `src/boundary_conditions/numerical/ghost.jl:98`. The
   latest commit is a Dependabot bump to 0.6 that looks unreconciled. **Deck slides 6,
   9 and 11 all depend on this running.**
2. **READMEs still say `Pkg.add(url=...)`** though all three packages are registered.
   O1 claims "registered" — make it true first.
3. **Don't put the README quick-start on a slide.** Its `rectangle(1m, 1m)` is a
   doc-local helper, not package API, so it won't run if someone copies it. The STL
   path (`import_mesh`) is real API and the better story anyway.
4. **The Niederer timings are >93% compilation time.** Use that benchmark as
   *validation* — it hits the reference activation times at all nine points — never as
   a speed claim. A speed claim also cuts against our own guidelines.
5. **The Niederer benchmark doesn't call Macchiato** — it uses RadialBasisFunctions.jl
   and WhatsThePoint directly. Fine given the ecosystem framing, but say "the
   ecosystem," not "Macchiato," when showing it.

---

## Still to sketch

Blocks 2–5 — API (slides 6–7), ecosystem table (8), WhatsThePoint (9),
RadialBasisFunctions (10), the two examples (11–12), conclusion (13).

Material already found that belongs to those blocks:

- **Slide 7 ("write your equations as they appear in textbook") has its artifact
  already built** — RadialBasisFunctions.jl's `@operator` macro:
  ```julia
  advdiff   = (@operator ν * ∇² - c ⋅ ∇)(x)
  diffusion = (@operator ∇ ⋅ (κ * ∇))(x)
  helmholtz = (@operator ∇² + k^2 * f)(x)
  ```
- **Slide 10 "order of convergence" is already done** — the generated convergence and
  work-precision study shipped in RBF.jl's docs, with CSVs, ~40 plots, and a recorded
  machine spec.
- **Slide 13 goals** (moved out of the Overview): complete incompressible flow · 3D
  and plane-strain elasticity · multi-domain and material interfaces · adaptive
  refinement · a symbolic PDE front-end.
