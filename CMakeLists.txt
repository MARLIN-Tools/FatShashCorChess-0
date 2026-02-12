cmake_minimum_required(VERSION 3.20)
project(Makaira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MAKAIRA_BUILD_TESTS "Build Makaira test executables" ON)

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

if (MAKAIRA_BUILD_TESTS)
    add_executable(makaira_perft_tests
        tests/perft_tests.cpp
    )
    target_link_libraries(makaira_perft_tests PRIVATE makaira_core)

    add_executable(makaira_search_tests
        tests/search_tests.cpp
    )
    target_link_libraries(makaira_search_tests PRIVATE makaira_core)

    add_executable(makaira_eval_tests
        tests/eval_tests.cpp
    )
    target_link_libraries(makaira_eval_tests PRIVATE makaira_core)
endif()

add_executable(makaira_eval_extract
    src/eval_extract.cpp
)
target_link_libraries(makaira_eval_extract PRIVATE makaira_core)

add_executable(makaira_eval_bench
    src/eval_bench.cpp
)
target_link_libraries(makaira_eval_bench PRIVATE makaira_core)

add_executable(makaira_search_bench
    src/search_bench.cpp
)
target_link_libraries(makaira_search_bench PRIVATE makaira_core)
