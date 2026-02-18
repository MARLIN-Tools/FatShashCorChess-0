# Makaira

Makaira is a C++ UCI chess engine using a classical alpha-beta search with a handcrafted evaluator.

## Build (Windows, CMake)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release --target makaira
```

## Run

```powershell
.\build\Release\makaira.exe
```

The binary starts in UCI mode. Typical smoke test input is:

```text
uci
isready
quit
```

## Repository Scope

This repository is intentionally trimmed to engine runtime/build essentials only.
