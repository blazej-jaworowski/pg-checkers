cmake_minimum_required(VERSION 3.8)
project(pg_monte_carlo_checkers CUDA)

add_executable(main main.cu Window.cu Window.cuh Engine.cu Engine.cuh GameState.cu GameState.cuh Node.cu Node.cuh)
target_link_libraries(main SDL2)

configure_file(chessboard.bmp chessboard.bmp COPYONLY)
configure_file(white_man.bmp white_man.bmp COPYONLY)
configure_file(white_king.bmp white_king.bmp COPYONLY)
configure_file(black_man.bmp black_man.bmp COPYONLY)
configure_file(black_king.bmp black_king.bmp COPYONLY)