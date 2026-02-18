cmake_minimum_required(VERSION 3.20)
project(FatShashCorChess0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fatshashcorchess0_core
    src/bitboard.cpp
    src/evaluator.cpp
    src/eval_tables.cpp
    src/hce_evaluator.cpp
    src/pawn_hash.cpp
    src/zobrist.cpp
    src/position.cpp
    src/movegen.cpp
    src/movepicker.cpp
    src/see.cpp
    src/perft.cpp
    src/search.cpp
)

target_include_directories(fatshashcorchess0_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (MSVC)
    target_compile_options(fatshashcorchess0_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(fatshashcorchess0_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(fatshashcorchess0
    src/main.cpp
)
target_link_libraries(fatshashcorchess0 PRIVATE fatshashcorchess0_core)
