rustsolve and rustmesh for the desktop Full build
================================================

pgcore, pyGIMLi's compiled core, solves its sparse systems with SuiteSparse
CHOLMOD and UMFPACK, and makes its 2-D meshes with Triangle. UMFPACK, and
CHOLMOD's Supernodal, Modify and MatrixOps modules, are GPL-2.0-or-later;
Triangle may not be distributed in a commercial product without its author's
permission. The desktop Full build replaces them with prebuilt files from this
folder:

  windows/libgimli.dll              pyGIMLi's core library rebuilt with rustsolve
                                    and without Triangle, for pgcore 1.5.0
                                    (gimli channel, py311_1)
  windows/rustsolve_capi.dll        rustsolve's C interface
  windows/rustmesh/_rustmesh.pyd    rustmesh's extension module
  macos/libgimli.dylib              the same for pgcore 1.5.4
                                    (np22py311h3a9382f_0), arm64
  macos/librustsolve_capi.dylib     rustsolve's C interface, arm64
  macos/rustmesh/_rustmesh.abi3.so  rustmesh's extension module, arm64
  rustmesh-package/*.py             rustmesh's Python package (both platforms)

rustsolve (https://github.com/geohang/rustsolve) is a sparse direct solver
library on faer; rustmesh (https://github.com/geohang/rustmesh) is a 2-D and
3-D mesh generator. Both are MIT OR Apache-2.0. PyHydroGeophysX's rustmesh
backend (PyHydroGeophysX/core/rustmesh_backend.py) routes pyGIMLi's 2-D
meshing to rustmesh; packaging/rthook_rustmesh.py turns it on in the built
program.

manifest.json records the files' SHA-256, the pgcore build each was verified
with, where they go and (Windows) the SuiteSparse DLLs pgcore carries that
they make redundant. apply.py installs them into the build environment and
checks the result; verify_core.py compares the rebuilt core with the original
in the same environment; packaging/release_audit.py lists them in
BUILD_INFO.txt with their notices from licenses/ (see THIRD_PARTY_NOTICES.md,
"Full builds: rustsolve and rustmesh in place of SuiteSparse and Triangle").

Where they come from
--------------------
rustsolve commit c83f4e9: the C interface, built with
"cargo build --locked --release -p rustsolve-capi --features nd", and the
gimli patch (pgcore/apply_rustsolve.py, including the change that takes the
primary potentials at the P2 nodes). apply_notriangle.py in this folder then
builds gimli without Triangle. rustmesh commit e332d94, built with
"maturin build --release --locked" (abi3, Python 3.10 and later).

Windows, on the developer's machine:
  libgimli.dll     gimli commit 5991b6ac (the source of pgcore 1.5.0) with
                   both patches, MSYS2 MINGW64 GCC 16.2.0, by rustsolve's
                   pgcore/build-libgimli-msys2.sh
  rustsolve_capi.dll, _rustmesh.pyd
                   rustc 1.99.0 (b940084d7), x86_64-pc-windows-gnu;
                   maturin 1.15.0
macOS, in GitHub Actions run 39815917732 of geohang/PyHydroGeophysX
(.github/workflows/build-rust-engines.yml), which uploaded them:
  libgimli.dylib   gimli v1.5.4 (6241cfab) with both patches, conda-forge
                   clang, by rustsolve's pgcore/build-libgimli-macos.sh
  librustsolve_capi.dylib, _rustmesh.abi3.so
                   rustc stable, aarch64-apple-darwin
The SHA-256 of each file is in manifest.json.

Rustsolve's build scripts also compile Triangle into their work folder;
apply_notriangle.py makes gimli's CMake ignore it, so libgimli neither
contains nor calls it.

Checks
------
On both platforms the rebuilt core passes rustsolve's C++ check of the
patched LinSolver (pgcore/check/) and verify_core.py: against the original
core, line 2's responses and Jacobians agree within the bounds there; its
primary potentials equal the P2 node values bit for bit; two runs in new
processes are bit-identical. The desktop Full build runs apply.py check,
verify_core.py and the built program's rustmesh self-test again on every
build.

Updating them
-------------
Replace the files, then update manifest.json (SHA-256, pgcore build and
MD5, the version string the core reports) and, when the crates change,
licenses/crates/ and each component's notices there. The crate list for a
build is
"cargo tree --locked -p rustsolve-capi --features nd -e normal,no-proc-macro
--target <target>" (rustsolve) and "cargo tree --locked -p rustmesh-py -e
normal,no-proc-macro --target <target>" (rustmesh).

licenses/
---------
rustsolve/                 rustsolve's licence texts and CONTENTS.txt
rustmesh/                  rustmesh's licence texts, spade's, and CONTENTS.txt
crates/<crate>-<version>/  each crate's licence files, as published
                           (SOURCE.txt where the crate ships none and the
                           text is from its repository)
rust-standard-library/     the Rust standard library's notices
pygimli/                   pyGIMLi's licence and MODIFICATIONS.txt
mingw-w64/                 the MinGW-w64 runtime licence (Windows files)
