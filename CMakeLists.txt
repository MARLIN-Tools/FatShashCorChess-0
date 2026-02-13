cmake_minimum_required(VERSION 3.20)
project(FatShashCorChess0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FATSHASHCORCHESS0_BUILD_TESTS "Build FatShashCorChess 0 test executables" ON)

find_package(ZLIB QUIET)

add_library(fatshashcorchess0_core
    src/bitboard.cpp
    src/evaluator.cpp
    src/eval_tables.cpp
    src/hybrid_evaluator.cpp
    src/hce_evaluator.cpp
    src/lc0/lc0_activation.cpp
    src/lc0/lc0_attention_value.cpp
    src/lc0/lc0_evaluator.cpp
    src/lc0/lc0_features112.cpp
    src/lc0/lc0_proto_wire.cpp
    src/lc0/lc0_tables.cpp
    src/lc0/lc0_weights.cpp
    src/pawn_hash.cpp
    src/zobrist.cpp
    src/position.cpp
    src/movegen.cpp
    src/movepicker.cpp
    src/perft.cpp
    src/search.cpp
)

target_include_directories(fatshashcorchess0_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (MSVC)
    target_compile_options(fatshashcorchess0_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(fatshashcorchess0_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (ZLIB_FOUND)
    target_compile_definitions(fatshashcorchess0_core PRIVATE MAKAIRA_HAS_ZLIB=1)
    target_link_libraries(fatshashcorchess0_core PRIVATE ZLIB::ZLIB)
else()
    target_compile_definitions(fatshashcorchess0_core PRIVATE MAKAIRA_HAS_ZLIB=0)
endif()

add_executable(fatshashcorchess0
    src/main.cpp
)
target_link_libraries(fatshashcorchess0 PRIVATE fatshashcorchess0_core)

if (FATSHASHCORCHESS0_BUILD_TESTS)
    add_executable(fatshashcorchess0_perft_tests
        tests/perft_tests.cpp
    )
    target_link_libraries(fatshashcorchess0_perft_tests PRIVATE fatshashcorchess0_core)

    add_executable(fatshashcorchess0_search_tests
        tests/search_tests.cpp
    )
    target_link_libraries(fatshashcorchess0_search_tests PRIVATE fatshashcorchess0_core)

    add_executable(fatshashcorchess0_eval_tests
        tests/eval_tests.cpp
    )
    target_link_libraries(fatshashcorchess0_eval_tests PRIVATE fatshashcorchess0_core)

    add_executable(fatshashcorchess0_lc0_loader_tests
        tests/lc0_loader_tests.cpp
    )
    target_link_libraries(fatshashcorchess0_lc0_loader_tests PRIVATE fatshashcorchess0_core)

    add_executable(fatshashcorchess0_lc0_forward_tests
        tests/lc0_forward_tests.cpp
    )
    target_link_libraries(fatshashcorchess0_lc0_forward_tests PRIVATE fatshashcorchess0_core)

    add_executable(fatshashcorchess0_lc0_toggle_tests
        tests/lc0_toggle_tests.cpp
    )
    target_link_libraries(fatshashcorchess0_lc0_toggle_tests PRIVATE fatshashcorchess0_core)
endif()

add_executable(fatshashcorchess0_eval_extract
    src/eval_extract.cpp
)
target_link_libraries(fatshashcorchess0_eval_extract PRIVATE fatshashcorchess0_core)

add_executable(fatshashcorchess0_eval_bench
    src/eval_bench.cpp
)
target_link_libraries(fatshashcorchess0_eval_bench PRIVATE fatshashcorchess0_core)

add_executable(fatshashcorchess0_search_bench
    src/search_bench.cpp
)
target_link_libraries(fatshashcorchess0_search_bench PRIVATE fatshashcorchess0_core)
