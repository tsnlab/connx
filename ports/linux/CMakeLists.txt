cmake_minimum_required(VERSION 3.10)
project(connx)

# variables
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CONNX_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(INFILE "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${INFILE}")
# list up generated source files using gen.sh utility
execute_process(
    COMMAND ../../bin/gen.sh -d -o ${CMAKE_CURRENT_BINARY_DIR}/src
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SRC)
string(REGEX REPLACE "\n$" "" SRC ${SRC})
string(PREPEND SRC "src/")
string(REPLACE "\n" ";src/" SRC ${SRC})

# list up original source files
file(GLOB_RECURSE CONNX_HDR "${CONNX_HOME}/include/*")
file(GLOB_RECURSE CONNX_SRC "${CONNX_HOME}/src/*")
file(GLOB_RECURSE HAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/opset.txt)
    set(OPSET ${CMAKE_CURRENT_SOURCE_DIR}/opset.txt)
else()
    set(OPSET "")
endif()

# generate source files
add_custom_command(
    OUTPUT ${SRC}
    COMMAND ../../bin/gen.sh -o ${CMAKE_CURRENT_BINARY_DIR}/src
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CONNX_SRC} ${CONNX_HDR} ${HAL_SRC} ${OPSET}
    COMMENT "Generating source code"
)

# compile
set(CMAKE_C_FLAGS_DEBUG "-pg -O0 -g -DDEBUG=1 -fsanitize=address")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

option(ENABLE_AVX "Enable/Disable AVX" OFF)
if(ENABLE_AVX)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -mavx")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx")
    message(STATUS "Enabling AVX")
endif()

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PUBLIC ${CONNX_HOME}/include)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Werror -std=c99)
target_link_libraries(${PROJECT_NAME} m pthread)

# Test
add_custom_target(onnx
    COMMAND python3 ${CONNX_HOME}/bin/test.py ${CMAKE_CURRENT_BINARY_DIR}/connx ${CONNX_HOME}
    DEPENDS ${PROJECT_NAME})
add_custom_target(mnist
    COMMAND python3 ${CONNX_HOME}/bin/run.py ${CMAKE_CURRENT_BINARY_DIR}/connx ${CONNX_HOME}/examples/mnist/ ${CONNX_HOME}/examples/mnist/test_data_set_0/input_0.data
    DEPENDS ${PROJECT_NAME})
add_custom_target(mobilenet
    COMMAND python3 ${CONNX_HOME}/bin/run.py ${CMAKE_CURRENT_BINARY_DIR}/connx ${CONNX_HOME}/examples/mobilenet/ ${CONNX_HOME}/examples/mobilenet/test_data_set_0/input_0.data
    DEPENDS ${PROJECT_NAME})
add_custom_target(prof
    COMMAND gprof ${CMAKE_CURRENT_BINARY_DIR}/connx ${CMAKE_CURRENT_BINARY_DIR}/gmon.out
    DEPENDS ${PROJECT_NAME})
