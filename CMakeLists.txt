cmake_minimum_required(VERSION 3.26)
project(hpc_praktikum C)

set(CMAKE_C_STANDARD 11)

add_executable(hpc_praktikum
        matrix_mul.c
        functions.c
        functions.h)

add_executable(tests
        tests/tests.c
        functions.c
        functions.h)
