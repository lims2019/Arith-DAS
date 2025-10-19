cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)
project(verilate)

message(
    STATUS
    "CMake start"
)

find_package(verilator HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
if(NOT verilator_FOUND)
    message(
        FATAL_ERROR
        "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable"
    )
else()
    message(STATUS "Verilator was found.")
endif()


add_executable(mul verilate/mul_verilate.cpp)

target_compile_features(mul PUBLIC cxx_std_14)

verilate(mul
    INCLUDE_DIRS "verilate"
    SOURCES path/to/MUL.v
    TOP_MODULE MUL
)

add_executable(add verilate/add_verilate.cpp)

target_compile_features(add PUBLIC cxx_std_14)

verilate(add
    INCLUDE_DIRS "verilate"
    SOURCES path/to/PrefixAdder.v
    TOP_MODULE PrefixAdder
)

add_executable(mac verilate/mac_verilate.cpp)

target_compile_features(mac PUBLIC cxx_std_14)

verilate(mac
    INCLUDE_DIRS "verilate"
    SOURCES path/to/MAC.v
    TOP_MODULE MUL
)
