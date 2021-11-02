cmake_minimum_required(VERSION 3.20)
project(pg_monte_carlo_checkers CUDA)

set(CMAKE_CUDA_STANDARD 14)


add_executable(pg_monte_carlo_checkers main.cu Window.cu Window.cuh Engine.cu Engine.cuh)
target_link_libraries(pg_monte_carlo_checkers SDL2)

configure_file(chessboard.bmp chessboard.bmp COPYONLY)
configure_file(white.bmp white.bmp COPYONLY)
configure_file(black.bmp black.bmp COPYONLY)

set_target_properties(pg_monte_carlo_checkers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
