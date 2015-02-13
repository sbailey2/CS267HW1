# on Hopper, we will benchmark you against Cray LibSci, the default vendor-tuned BLAS. The Cray compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. On Hopper, the Portland compilers are default, so you must instruct the Cray compiler wrappers to switch to GNU: type "module swap PrgEnv-pgi PrgEnv-gnu"

CC = cc 
#OPT = -g
OPT = -O3 -march=native
#OPT =  
CFLAGS = -Wall -std=gnu99 $(OPT)
#CFLAGS = $(OPT)
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt 
#LDLIBS = 

#targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-packed benchmark-packed-intrinsics benchmark-packed2 benchmark-mlblocked
#objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-packed.o dgemm-packed-intrinsics.o dgemm-packed2.o dgemm-mlblocked.o
targets = benchmark-intrinsics3
objects = benchmark.o dgemm-intrinsics3.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

#benchmark-naive : benchmark.o dgemm-naive.o 
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-blocked : benchmark.o dgemm-blocked.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-packed-intrinsics : benchmark.o dgemm-packed-intrinsics.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-packed : benchmark.o dgemm-packed.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-blas : benchmark.o dgemm-blas.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-packed2 : benchmark.o dgemm-packed2.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-mlblocked : benchmark.o dgemm-mlblocked.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-intrinsics2 : benchmark.o dgemm-intrinsics2.o
#	$(CC) -o $@ $^ $(LDLIBS)
benchmark-intrinsics3 : benchmark.o dgemm-intrinsics3.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
