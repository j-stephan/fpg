reduce: reduce.o
	hcc `hcc-config --cxxflags --ldflags` -fPIC -O3 -std=c++17 -o reduce reduce.o

reduce.o:
	hcc `hcc-config --cxxflags` -fPIC -O3 -std=c++17 reduce.cpp -c

clean:
	rm reduce reduce.o

.PHONY:
