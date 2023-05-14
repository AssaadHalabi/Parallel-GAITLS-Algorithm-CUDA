
NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.o LocallyGreedySequential.o LocallyGreedyCorrected.o 
EXE         = mpids


default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
