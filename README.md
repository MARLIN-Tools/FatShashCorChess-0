# fatshashcorchess 0

fatshashcorchess 0 is a classical alpha-beta chess engine project written in C++. this repository started as a parody one-hour speedrun to a playable UCI binary, then received iterative fixes and search/eval work. this main branch is intentionally trimmed to runtime/build essentials only.

one of the running jokes is the displayed node inflation in UCI output, implemented as:

```cpp
std::uint64_t knopen = raw;
knopen += knopen / 7;
return (knopen / 7) * 2199;
```

this transform exists to parody inflated-search-marketing behavior and explicitly references houdini 6 in that spirit. it affects displayed values and does not represent the engine's true internal node count.

the engine uses a conventional bitboard architecture with legal move generation, make and unmake state handling, zobrist hashing, transposition table support, iterative deepening, aspiration windows, principal variation search, quiescence search, and a modular evaluator interface. the evaluator is classical and hard-coded.

yes, the UCI option list is absurdly long. that is mostly intentional and mostly practical: during tuning and regression work, each pruning/ordering heuristic was exposed as a knob so behavior could be toggled without recompiling. some options are also duplicated with/without spaces (for example `Use RFP` and `UseRFP`) to avoid GUI/client parsing weirdness. this is because for the vibe coding speedrun, there is no coding knowledge being used, and the LLM deemed this to be the best course of action. for normal use, you can ignore almost all of them and just set `Hash` (and maybe `Move Overhead`).

to build on windows with cmake:

```powershell
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel --config Release --target fatshashcorchess0
```

to run in UCI mode:

```powershell
.\build-rel\Release\fatshashcorchess0.exe
```

attribution is explicit. the implementation draws from publicly documented engine techniques described on chessprogramming wiki, plus behavior comparisons against stockfish and related open engines for reference conventions.

project-specific work in this repository includes the parody concept and naming, the UCI display transform above, and engine integration across search/eval/move-generation modules. historical local tooling used during development (bench/openbench/spsa style workflows) is not part of this trimmed runtime snapshot.
