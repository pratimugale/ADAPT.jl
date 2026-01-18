# Variables
JULIA = julia

.PHONY: all install test clean

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