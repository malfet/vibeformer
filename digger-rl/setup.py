import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _build_mode() -> str:
    """Match DOSBox Pure's BUILD modes so both halves stay consistent.

    Priority: BUILD (explicit) > DEBUG=1 > RELEASEDBG=1 > RELEASE.
    """
    explicit = os.environ.get("BUILD")
    if explicit:
        return explicit.upper()
    if os.environ.get("DEBUG") == "1":
        return "DEBUG"
    if os.environ.get("RELEASEDBG") == "1":
        return "RELEASEDBG"
    return "RELEASE"


_FLAGS = {
    "DEBUG":      (["-O0", "-g", "-DDEBUG"],            ["-g"]),
    "RELEASEDBG": (["-O2", "-g", "-DNDEBUG"],           ["-g"]),
    "PROFILE":    (["-O2", "-DNDEBUG"],                 []),
    "ASAN":       (["-O0", "-g", "-fsanitize=address",
                    "-fno-omit-frame-pointer", "-DDEBUG"],
                   ["-fsanitize=address"]),
    "RELEASE":    (["-O3", "-DNDEBUG"],                 []),
}

_mode = _build_mode()
extra_compile, extra_link = _FLAGS.get(_mode, _FLAGS["RELEASE"])
print(f"setup.py: BUILD={_mode} CXXFLAGS+={extra_compile} LDFLAGS+={extra_link}")

ext = Pybind11Extension(
    "_libretro",
    ["csrc/libretro_frontend.cpp"],
    include_dirs=["vendor/dosbox-pure/libretro-common/include"],
    cxx_std=17,
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
