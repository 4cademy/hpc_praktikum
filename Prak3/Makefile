SRC = functions.cu matrix_mul.cu
OBJ = functions.o matrix_mul.o
TGT = matmul
DEP = my_c_file.d my_cu_file.d

$(TGT): functions.o matrix_mul.o
	nvcc -O3 -o program functions.o matrix_mul.o -Xcompiler --std=c++11

functions.o: functions.cu
	nvcc -c functions.cu -Xcompiler --std=c++11 -lcuda -lcudart

matrix_mul.o: matrix_mul.cu
	nvcc -c matrix_mul.cu -Xcompiler --std=c++11 -lcuda -lcudart 

clean:
	rm -f $(OBJ) $(TGT) $(DEP)

-include $(DEP)