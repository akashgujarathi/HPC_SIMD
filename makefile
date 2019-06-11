all: 
	g++ -mavx2 -o assgn_2_2 simd.cpp
	@echo Object ./assgn_2_2
clean:
	rm assgn_2_2
