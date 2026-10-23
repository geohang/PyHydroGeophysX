# spade, vendored

This is spade 2.15.1 by Stefan Altmayer (https://github.com/Stoeoef/spade),
MIT OR Apache-2.0 (see `LICENSE-MIT` and `LICENSE-APACHE` here), copied from
the crates.io release. `src/` and `images/` are unchanged except for the
patches below, which are all in `src/delaunay_core/refinement.rs`; the file
says so in its first line, as the Apache-2.0 licence asks of modified files.
The original examples, benchmarks and dev-dependencies are left out
(`Cargo.toml` is trimmed accordingly).

The package is named `rustmesh-spade`, version `2.15.1-rustmesh.1`. The library
name stays `spade`, so source that uses spade compiles unchanged. rustmesh
depends on it by package alias and exact version, so that a downstream crate
cannot pick up the unpatched crates.io spade by accident. The fork has to be
published before rustmesh is.

To update spade, copy the new release's `src/` and `images/`, re-apply the
edits below (search for "rustmesh patch" and "rustmesh:"), and run
`cargo test -p rustmesh`.

## Patch: `refine_with`

- New public `refine_with(parameters, excluded_faces, max_area)`: refinement
  with the caller's own excluded faces instead of the set
  `exclude_outer_faces` computes, and a per-face area limit
  `Fn(&[Point2; 3]) -> Option<S>`.
- `refine` calls a private `refine_impl`, which is the original body with two
  changes: the initial excluded set may be given, and the area limit passed to
  `get_refinement_hint` is the smaller of the face's own limit and the global
  one.
- `get_refinement_hint` takes the area limit as an argument.

Why: spade's `calculate_outer_faces` peels constraint-separated layers in
stack order. In pyGIMLi's ERT parameter-mesh layout the parameter domain is
one constraint from outside (through the ground surface) and two through the
outer region, and the peel can put it in an even, excluded layer. rustmesh
therefore classifies faces itself (seed flood fill) and needs a way to hand
spade that classification, plus region-dependent cell sizes.

## Patch: `refine_with_control`

`refine_with_control` adds a callback before each refinement queue iteration.
Returning false stops further insertion and marks refinement incomplete. The
callback sees all working vertices and inner faces, including excluded faces.
`refine` and `refine_with` use the same implementation without a callback.

## Patch: exact `nearest_power_of_two`

`input.log2().round().exp2()` is replaced by doubling and halving, and one
squared comparison against 2. The results are the same (checked on 449 313
values from 1e-12 to 1e12 and on spade's own test cases), but exact: the last
bits of `log2` differ between math libraries, so a WebAssembly build split
segments at slightly different points than a native build and the same model
meshed differently.

## Patch: split the edge in front of a too-large cell's circumcenter

When the circumcenter of a face that is larger than its area limit lies in an
excluded face (air, a hole) or outside the convex hull, spade skipped the face
and never came back to it, so the mesh kept a cell over its limit (1 to 2 % of
random geometries; the
`a_cell_whose_circumcentre_lies_in_a_hole_still_meets_its_area_limit` test in
`crates/rustmesh/tests/mesh2d_random.rs` is one). The face now pushes the first
fixed edge between it and its circumcenter (`fixed_edge_towards`, a walk with
spade's `LineIntersectionIterator`) to the forced-split buffer and is revisited,
as in Ruppert's algorithm. A face that is only skinny is still skipped there:
doing the same for it ran away near small input angles (one random case spent a
100 000-vertex budget on slivers).

## Patch: kept edges, `refine_with_kept_edges`

- New public `refine_with_kept_edges(parameters, excluded_faces, kept_edges,
  max_area, checkpoint)`: `refine_with_control` plus a list of constraint
  edges, each named by its two vertices, that refinement never splits.
  `refine_impl` takes the set (empty for the other entry points) and checks it
  with `is_kept` at four places: an encroached segment taken from the queue, a
  circumcenter lying on an edge, the edge `fixed_edge_towards` finds in front of
  a circumcenter in an excluded face or outside the hull, and an encroached
  fixed edge met while simulating the insertion. A kept edge is never queued
  for splitting, and a circumcenter that would encroach, lie on or lie beyond
  it is not inserted. This is the rule `keep_constraint_edges` applies to every
  constraint edge, here applied per edge.
- Kept edges are never split, so their vertex handles name them for the whole
  refinement.

Why: pyGIMLi's `createMesh(..., preserveBoundary=...)` passes Triangle's `-Y`
switch (no Steiner point on boundary segments), and `appendTriangleBoundary`
needs it so that the appended mesh meets the original mesh node for node.
rustmesh's `Options::keep_segments` and `keep_boundary` select the edges. Faces
next to a kept edge may stay below the angle target or above the area target.

## Patch: scheduling and off-centers above 30 degrees

For angle targets above 30 degrees, refinement takes the faces with the
shortest edge first, keeps a snapshot of each queued face's vertices so that
stale entries are discarded, and breaks ties by insertion order. Encroached
segments keep priority. An off-center on the shortest edge's bisector (Üngör,
2004 and 2009) may replace the circumcenter; it goes through the same
constraint and kept-edge checks. Up to 30 degrees the queue order is unchanged.
This is not a full implementation of Chew's second algorithm or of Shewchuk's
Terminator, and it does not guarantee a 34 degree minimum angle.

## Patch: refinement queue and corner handling

Above 30 degrees the queues hold only bad faces, in narrow length bins taken
shortest bin first, first in first out within a bin. Cached hints are valid
while the triangle's vertices are unchanged, and size callbacks are treated as
functions of the supplied positions. Excluded faces use indexed flags. A
candidate insertion reuses the located face or edge and the robust incircle flip
plan already found while simulating the insertion, and still validates the
vertex and notifies the hint generators.

At the corner between two boundary segments, the ancestry guard is relaxed so
that refinement can move away from the corner, while the short edge across the
corner may not force another split of a segment still incident to it. Internal
interfaces keep the conservative guard. Each skipped face is returned as a
snapshot of its vertices with a reason (for example `input_corner_triangle`,
`shared_input_corner_guard`, `kept_constraint`); these are diagnostics, not
quality exemptions. The flip traversal and the boundary relaxation can also
change meshes at 30 degrees and below.
