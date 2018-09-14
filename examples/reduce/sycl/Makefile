SYCL_INCDIR=/opt/sycl/computecpp/include
SYCL_LIBDIR=/opt/sycl/computecpp/lib
CL_INCDIR=/opt/rocm/opencl/include

reduce-ptx: reduce-ptx.o
	compute++ -sycl-driver -sycl-target ptx64 -O3 -std=c++17 -o reduce-ptx \
		reduce-ptx.o -L${SYCL_LIBDIR} -lComputeCpp

reduce-ptx.o:
	compute++ -sycl-driver -sycl-target ptx64 -O3 -std=c++17 reduce.cpp -c \
		-I${SYCL_INCDIR} -I${CL_INCDIR} -o reduce-ptx.o

reduce-spir: reduce-spir.o
	compute++ -sycl-driver -sycl-target spir64 -O3 -std=c++17 -o reduce-spir \
		reduce-spir.o -L${SYCL_LIBDIR} -lComputeCpp

reduce-spir.o:
	compute++ -sycl-driver -sycl-target spir64 -O3 -std=c++17 reduce.cpp -c \
		-I${SYCL_INCDIR} -I${CL_INCDIR} -o reduce-spir.o

reduce-spirv: reduce-spirv.o
	compute++ -sycl-driver -sycl-target spirv64 -O3 -std=c++17 -o reduce-spirv \
		reduce-spirv.o -L${SYCL_LIBDIR} -lComputeCpp

reduce-spirv.o:
	compute++ -sycl-driver -sycl-target spirv64 -O3 -std=c++17 reduce.cpp -c \
		-I${SYCL_INCDIR} -I${CL_INCDIR} -o reduce-spirv.o

clean:
	rm -f reduce-spir reduce-spir.o \
	   	  reduce-ptx reduce-ptx.o \
	   	  reduce-spirv reduce-spirv.o

.PHONY:
