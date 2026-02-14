param(
    [string]$BenchExe = ".\build-rel\Release\fatshashcorchess0_search_bench.exe",
    [string]$FenSuite = ".\bench\fens.txt",
    [int]$Depth = 10,
    [UInt64]$Nodes = 0,
    [int]$HashMB = 32,
    [string]$Output = ".\fatshashcorchess0_cpu.etl"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $BenchExe)) {
    throw "Bench executable not found: $BenchExe"
}

if (!(Test-Path $FenSuite)) {
    throw "FEN suite not found: $FenSuite"
}

$args = @("--fen-suite", $FenSuite, "--depth", "$Depth", "--hash", "$HashMB")
if ($Nodes -gt 0) {
    $args += @("--nodes", "$Nodes")
}

Write-Host "Starting WPR CPU trace..."
wpr -start CPU -filemode | Out-Null

try {
    Write-Host "Running bench: $BenchExe $($args -join ' ')"
    & $BenchExe @args
}
finally {
    Write-Host "Stopping WPR trace: $Output"
    wpr -stop $Output | Out-Null
}

Write-Host "Done. Open $Output in Windows Performance Analyzer."
