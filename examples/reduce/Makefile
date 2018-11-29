CXX = nvcc
CXXFLAGS = -O3 -std=c++14 --disable-warnings -m 64 -arch sm_70 \
		   -Xcompiler -Wall,-Wextra,-pedantic
LDFLAGS = -O3

reduce: accelerator.o benchmark.o main.o
	$(CXX) -o reduce accelerator.o benchmark.o main.o $(LDFLAGS)

accelerator.o: accelerator.cu accelerator.h
	$(CXX) $(CXXFLAGS) accelerator.cu -c

benchmark.o: benchmark.cpp benchmark.h
	$(CXX) $(CXXFLAGS) benchmark.cpp -c

main.o: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -c

clean:
	rm -f reduce *.o

.PHONY:
