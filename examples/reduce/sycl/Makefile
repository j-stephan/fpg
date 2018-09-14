SYCL_INCDIR=/home/jan/software/computecpp/include
SYCL_LIBDIR=/home/jan/software/computecpp/lib

reduce: reduce.o
	compute++ -sycl-driver -sycl-target ptx64 -O3 -std=c++17 -o reduce reduce.o -L${SYCL_LIBDIR} -lComputeCpp

reduce.o:
	compute++ -sycl-driver -sycl-target ptx64 -O3 -std=c++17 reduce.cpp -c -I${SYCL_INCDIR}

clean:
	rm reduce reduce.o

.PHONY:
