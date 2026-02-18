cmake_minimum_required(VERSION 3.20)
project(Makaira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(makaira_core
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

target_include_directories(makaira_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (MSVC)
    target_compile_options(makaira_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(makaira_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(makaira
    src/main.cpp
)
target_link_libraries(makaira PRIVATE makaira_core)
