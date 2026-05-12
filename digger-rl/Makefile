# Top-level orchestrator: builds both the DOSBox Pure libretro core and the
# pybind11 extension with consistent flags. Build mode is picked from env:
#   DEBUG=1        -> -O0 -g everywhere
#   RELEASEDBG=1   -> -O2 -g everywhere (RelWithDebInfo)
#   BUILD=<MODE>   -> explicit override (RELEASE/DEBUG/RELEASEDBG/PROFILE/ASAN)
# Default is RELEASE.

ifdef BUILD
  MODE := $(BUILD)
else ifeq ($(DEBUG),1)
  MODE := DEBUG
else ifeq ($(RELEASEDBG),1)
  MODE := RELEASEDBG
else
  MODE := RELEASE
endif

DOSBOX_PURE := vendor/dosbox-pure
PIP         := .venv/bin/pip
NPROC       := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc)

# DOSBox Pure's Makefile only skips strip when BUILD=DEBUG; for RELEASEDBG it
# strips the dylib, throwing away symbols we explicitly asked to keep. Suppress
# strip for anything but a clean release build.
ifeq ($(MODE),RELEASE)
  STRIP_OVERRIDE :=
else
  STRIP_OVERRIDE := STRIP=true
endif

# DOSBox Pure's dynarec writes generated x86->arm64 code and then mprotects the
# region executable. On Apple Silicon this requires MAP_JIT + a JIT entitlement
# on the host process, neither of which DOSBox Pure (nor Homebrew Python) has,
# so the SIGILL we hit in cache_flush is the kernel rejecting the broken JIT
# state. Disable the dynarec and fall back to the interpreter cores -- still
# plenty fast for DOS-era games.
CXX_FOR_CORE := c++ -DDISABLE_DYNAREC=1

.PHONY: all core ext clean clean-core clean-ext debug relwithdebinfo release help

all: core ext
	@echo ">>> done (BUILD=$(MODE))"

core:
	@echo ">>> building DOSBox Pure libretro core (BUILD=$(MODE))"
	$(MAKE) -C $(DOSBOX_PURE) -j$(NPROC) BUILD=$(MODE) $(STRIP_OVERRIDE) \
	    CXX="$(CXX_FOR_CORE)"

ext:
	@echo ">>> installing dev deps from requirements.txt"
	$(PIP) install -r requirements.txt
	@echo ">>> building pybind11 extension (BUILD=$(MODE))"
	BUILD=$(MODE) $(PIP) install -e . --no-build-isolation

# Convenience aliases
debug:
	$(MAKE) DEBUG=1 all

relwithdebinfo:
	$(MAKE) RELEASEDBG=1 all

release:
	$(MAKE) all

clean: clean-core clean-ext

clean-core:
	$(MAKE) -C $(DOSBOX_PURE) clean

clean-ext:
	rm -rf build/ _libretro*.so digger_rl.egg-info/ *.egg-info/

help:
	@echo "Targets:"
	@echo "  make              -> release build of core + extension"
	@echo "  make debug        -> debug build of both"
	@echo "  make relwithdebinfo -> -O2 + debug symbols on both"
	@echo "  DEBUG=1 make      -> same as 'make debug'"
	@echo "  BUILD=ASAN make   -> AddressSanitizer build"
	@echo "  make clean        -> clean both"
