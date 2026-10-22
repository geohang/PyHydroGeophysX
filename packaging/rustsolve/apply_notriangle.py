"""Build a gimli 1.5.x tree without Triangle; run after rustsolve's apply_rustsolve.py.

usage: python apply_notriangle.py <gimli source dir>

CMake no longer looks for Triangle (TRIANGLE_FOUND is 0), so libgimli links no
Triangle code even when a build script compiles Triangle next to it. The
wrapper's Triangle calls then throw an error that names the rustmesh backend
(PyHydroGeophysX/core/rustmesh_backend.py) instead of printing "Triangle not
installed" and returning an empty mesh. pg.core.versionStr() gains
"+notriangle". Each changed file gets a notice under its licence header, as
Apache-2.0 asks of modified files. Anchored on lines that exist in gimli
v1.5.0-v1.5.5; refuses to run if an anchor is missing or the tree is already
patched.
"""
import pathlib
import sys

NOTICE = ("Modified 2026 by Hang Chen for PyHydroGeophysX: built without Triangle; "
          "pyGIMLi's 2-D meshes come from rustmesh.")

FIND_OLD = """find_or_build_package(Triangle triangle LOCAL)
"""
FIND_NEW = """# Built without Triangle: TRIANGLE_FOUND 0 keeps it out of libgimli.
set(TRIANGLE_FOUND 0)
"""
PYGIMLI_OLD = """if (Python_EXECUTABLE
    AND TRIANGLE_FOUND
"""
PYGIMLI_NEW = """string(REPLACE "\\"" "" PACKAGE_VERSION "${PACKAGE_VERSION}")
set(PACKAGE_VERSION \\"${PACKAGE_VERSION}+notriangle\\")

if (Python_EXECUTABLE
"""

NOT_INSTALLED = """    std::cerr << WHERE_AM_I << " Triangle not installed" << std::endl;
"""
REFUSE = """    throwError(WHERE_AM_I + " libgimli is built without Triangle; mesh with the rustmesh backend");
"""
WRAPPER = [
    # destructor and freeMemory_: nothing to free
    ("""    if (mesh_voronoi_output_) delete mesh_voronoi_output_;
#else
""" + NOT_INSTALLED, """    if (mesh_voronoi_output_) delete mesh_voronoi_output_;
"""),
    ("""    delete [] mesh_input_->regionlist;
#else
""" + NOT_INSTALLED, """    delete [] mesh_input_->regionlist;
"""),
    # init_: a wrapper can be made, it cannot mesh
    ("""    allocateOutMemory_();
#else
""" + NOT_INSTALLED, """    allocateOutMemory_();
#else
    mesh_input_ = mesh_output_ = mesh_voronoi_output_ = NULL;
"""),
    # generate, allocateOutMemory_, transformTriangleToMesh_, transformMeshToTriangle_
    ("""    transformTriangleToMesh_(* mesh_output_, mesh);
#else
""" + NOT_INSTALLED, """    transformTriangleToMesh_(* mesh_output_, mesh);
#else
""" + REFUSE),
    ("""    mesh_output_->normlist              = (REAL *) NULL;
#else
""" + NOT_INSTALLED, """    mesh_output_->normlist              = (REAL *) NULL;
#else
""" + REFUSE),
    ("""//   }
#else
""" + NOT_INSTALLED, """//   }
#else
""" + REFUSE),
    ("""        }
    }
#else
""" + NOT_INSTALLED, """        }
    }
#else
""" + REFUSE),
]


def add_notice(text):
    if text.startswith("/*"):
        end = text.index("\n", text.index("*/") + 2) + 1
        return text[:end] + "\n/*\n * " + NOTICE + "\n */\n" + text[end:]
    return "# " + NOTICE + "\n" + text


def edit(path, pairs):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    if "+notriangle" in text or "built without Triangle" in text:
        sys.exit(f"{path} is already patched")
    for old, new in pairs:
        if text.count(old) != 1:
            sys.exit(f"{path}: anchor found {text.count(old)} times, expected 1")
        text = text.replace(old, new)
    if "Triangle not installed" in text:
        sys.exit(f"{path}: a 'Triangle not installed' branch is left")
    text = add_notice(text)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def main():
    src = pathlib.Path(sys.argv[1]).resolve()
    top = src / "CMakeLists.txt"
    if "GIMLI_RUSTSOLVE" not in top.read_text(encoding="utf-8"):
        sys.exit(f"{src}: run rustsolve's apply_rustsolve.py first")
    edit(top, [(FIND_OLD, FIND_NEW), (PYGIMLI_OLD, PYGIMLI_NEW)])
    edit(src / "core" / "src" / "triangleWrapper.cpp", WRAPPER)
    print(f"patched {src}: no Triangle")


if __name__ == "__main__":
    main()
