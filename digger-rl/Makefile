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

.PHONY: all core ext clean clean-core clean-ext debug relwithdebinfo release help

all: core ext
	@echo ">>> done (BUILD=$(MODE))"

core:
	@echo ">>> building DOSBox Pure libretro core (BUILD=$(MODE))"
	$(MAKE) -C $(DOSBOX_PURE) -j$(NPROC) BUILD=$(MODE)

ext:
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
