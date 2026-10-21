rustsolve for the desktop Full build
====================================

pgcore, pyGIMLi's compiled core, solves its sparse systems with SuiteSparse
CHOLMOD and UMFPACK. UMFPACK, and CHOLMOD's Supernodal, Modify and MatrixOps
modules, are GPL-2.0-or-later. The desktop Full build puts rustsolve
(https://github.com/geohang/rustsolve, MIT OR Apache-2.0, a private
repository) in their place, from the prebuilt files in this folder:

  windows/libgimli.dll            pyGIMLi's core library rebuilt with rustsolve,
                                  for pgcore 1.5.0 (gimli channel, py311_1)
  windows/rustsolve_capi.dll      rustsolve's C interface
  macos/libgimli.dylib            the same for pgcore 1.5.4
                                  (np22py311h3a9382f_0), arm64
  macos/librustsolve_capi.dylib   rustsolve's C interface, arm64

manifest.json records their SHA-256, the pgcore build each was verified
with, where they go and (Windows) the SuiteSparse DLLs pgcore carries that
they make redundant. apply.py installs them into the build environment and
checks the result; packaging/release_audit.py lists them in BUILD_INFO.txt
with their notices from licenses/ (see THIRD_PARTY_NOTICES.md, "Full builds:
rustsolve in place of CHOLMOD and UMFPACK").

Where they come from
--------------------
rustsolve commit fb18324 (the library and the pgcore patch; unchanged
through 13a3502), C interface built with
"cargo build --locked --release -p rustsolve-capi --features nd".

windows/libgimli.dll     SHA-256 5eedb6339e391a337e0fc8f456b8ea4cc543eb6f57d54276eb876c92def4aede
    gimli commit 5991b6ac with pgcore/apply_rustsolve.py, MSYS2 MINGW64
    GCC 16.2.0 (pgcore/build-libgimli-msys2.sh), on the developer's machine.
windows/rustsolve_capi.dll  SHA-256 120bbe0ba5400271b15b4e4f2ceba2fe0bb2749543f7df4ad37c510d798e6bd1
    rustc 1.99.0, x86_64-pc-windows-gnu, on the developer's machine. A
    rebuild from 13a3502 differs from it only in the PE time stamps and
    checksum (9 bytes).
macos/libgimli.dylib     SHA-256 4c743324db353a6dfedd02b6d161ba30dbdaf113820554560962c9c31d944b5c
macos/librustsolve_capi.dylib  SHA-256 dfa55eaf20a36385d36339b0bca6d8e984af1d427e45987ff90b4ba90e81fe25
    gimli v1.5.4 (6241cfab) with pgcore/apply_rustsolve.py, conda-forge
    clang 21.1.8 (pgcore/build-libgimli-macos.sh); rustc 1.99.1,
    aarch64-apple-darwin; GitHub Actions run 39546992242 of
    geohang/rustsolve (pgcore-macos.yml), which uploaded them as an
    artifact.

These are the builds rustsolve's docs/pgcore-verification.md describes:
on Windows verify10 (2026-10-20), on macOS run 39546992242 (2026-10-21);
every check and gate there passed.

Updating them
-------------
Replace the files, then update manifest.json (SHA-256, pgcore build and
MD5, the version string the core reports) and, when the crates change,
licenses/crates/ and the list below. The crate list for a build is
"cargo tree --locked -p rustsolve-capi --features nd -e normal,no-proc-macro
--target <target>".

licenses/
---------
rustsolve/                 rustsolve's licence texts and CONTENTS.txt
crates/<crate>-<version>/  each crate's licence files, as published
                           (SOURCE.txt where the crate ships none and the
                           text is from its repository)
rust-standard-library/     the Rust standard library's notices (Rust
                           1.99.0 and 1.99.1 ship the same file)
pygimli/                   pyGIMLi's licence and MODIFICATIONS.txt
mingw-w64/                 the MinGW-w64 runtime licence (Windows files)

Rust crates compiled into rustsolve_capi
----------------------------------------
  atomic-wait 1.1.0             BSD-2-Clause                   both
  bitflags 2.13.2               MIT OR Apache-2.0              both
  bytemuck 1.25.2               Zlib OR Apache-2.0 OR MIT      both
  byteorder 1.5.0               Unlicense OR MIT               macOS
  cfg-if 1.0.5                  MIT OR Apache-2.0              both
  crossbeam 0.8.5               MIT OR Apache-2.0              both
  crossbeam-channel 0.5.17      MIT OR Apache-2.0              both
  crossbeam-deque 0.8.8         MIT OR Apache-2.0              both
  crossbeam-epoch 0.9.21        MIT OR Apache-2.0              both
  crossbeam-queue 0.3.14        MIT OR Apache-2.0              both
  crossbeam-utils 0.8.23        MIT OR Apache-2.0              both
  dyn-stack 0.13.2              MIT                            both
  either 1.19.0                 MIT OR Apache-2.0              both
  equator 0.2.2                 MIT                            both
  equator 0.4.2                 MIT                            both
  equator 0.6.0                 MIT                            both
  faer 0.24.4                   MIT                            both
  faer-traits 0.24.0            MIT                            both
  feral-amd 0.2.1               MIT                            both
  feral-metis 0.3.0             MIT                            both
  feral-ordering-core 0.2.2     MIT                            both
  gemm 0.19.0                   MIT                            both
  gemm-c32 0.19.0               MIT                            both
  gemm-c64 0.19.0               MIT                            both
  gemm-common 0.19.0            MIT                            both
  gemm-f32 0.19.0               MIT                            both
  gemm-f64 0.19.0               MIT                            both
  generativity 1.2.1            MIT OR Apache-2.0              both
  libc 0.2.191                  MIT OR Apache-2.0              macOS
  libm 0.2.16                   MIT                            both
  nano-gemm 0.2.2               MIT                            both
  nano-gemm-c32 0.2.1           MIT                            both
  nano-gemm-c64 0.2.1           MIT                            both
  nano-gemm-core 0.2.1          MIT                            both
  nano-gemm-f32 0.2.1           MIT                            both
  nano-gemm-f64 0.2.1           MIT                            both
  num-complex 0.4.6             MIT OR Apache-2.0              both
  num-traits 0.2.19             MIT OR Apache-2.0              both
  num_cpus 1.17.0               MIT OR Apache-2.0              Windows
  once_cell 1.21.4              MIT OR Apache-2.0              both
  private-gemm-x86 0.1.22       MIT                            Windows
  pulp 0.22.3                   MIT                            both
  pulp-wasm-simd-flag 0.1.1     MIT                            both
  qd 0.8.0                      MIT                            both
  raw-cpuid 11.6.0              MIT                            both
  rayon 1.12.0                  MIT OR Apache-2.0              both
  rayon-core 1.13.0             MIT OR Apache-2.0              both
  reborrow 0.5.5                MIT                            both
  spindle 0.2.6                 MIT                            both
  sysctl 0.6.0                  MIT                            macOS
  thiserror 1.0.69              MIT OR Apache-2.0              macOS
  windows-sys 0.42.0            MIT OR Apache-2.0              Windows
  windows_x86_64_gnu 0.42.2     MIT OR Apache-2.0              Windows
