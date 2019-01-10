CXX = compute++
SYCL_INC = /opt/sycl/computecpp/include
CL_INC = /home/jstephan/software/cuda/include
CXXFLAGS = -O3 -std=c++14 -Wall -Wextra -pedantic
SYCL_FLAGS = -I$(SYCL_INC) -I$(CL_INC) -sycl-driver -sycl-target ptx64 \
			 -no-serial-memop
LDFLAGS = -O3 -lComputeCpp -lOpenCL

reduce: accelerator.o benchmark.o main.o
	$(CXX) -o reduce accelerator.o benchmark.o main.o $(LDFLAGS)

accelerator.o: accelerator.cpp accelerator.h
	$(CXX) $(CXXFLAGS) $(SYCL_FLAGS) accelerator.cpp -c

benchmark.o: benchmark.cpp benchmark.h
	$(CXX) $(CXXFLAGS) benchmark.cpp -c

main.o: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -c

clean:
	rm -f reduce *.o

.PHONY:
