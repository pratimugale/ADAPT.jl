# Variables
JULIA = julia

.PHONY: all install test clean smoke smoke-kamis install-kamis clean-kamis

all: install

install:
	@echo "Initializing TetrisADAPT environment..."
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	pip install -r requirements.txt
	julia --project=. -e 'using Pkg; Pkg.build("PyCall")'

smoke:
	@echo "Running TetrisADAPT smoke test that runs 1 example of maxcut_qaoa with qiskit interface"
	$(JULIA) --project=. test/maxcut_qaoa.jl

clean:
	@echo "Removing Julia artifacts..."
	rm -rf ~/.julia/compiled/v1.*/TetrisADAPT

test:
	$(JULIA) --project=. -e 'using Pkg; Pkg.test()'

install-kamis:
	@echo "Installing KaMIS..."
	@mkdir -p external
	git clone https://github.com/pratimugale/KaMIS.git external/KaMIS
	cd external/KaMIS && git submodule update --init --recursive
	@echo "KaMIS repo cloned!"
	@echo "To compile KaMIS, run: cd external/KaMIS && ./compile_withcmake.sh"
	@echo "If using MacOS, you may need to edit the compile_withcmake.sh script to use the correct compiler. Please refer to external/KaMIS/HowToMacOs.md for instructions."
	@echo "To clean KaMIS, run: rm -rf external/KaMIS"

clean-kamis:
	rm -rf external/KaMIS

smoke-kamis:
	@echo "Running KaMIS mmwis smoke test..."
	$(JULIA) --project=. test/kamis_smoke_test.jl