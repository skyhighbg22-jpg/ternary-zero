# =====================================================================
# GEMV Benchmark Runner
# =====================================================================
# Compiles and runs the custom ternary GEMV kernel vs cuBLAS FP16
# benchmark, with optional Nsight Compute profiling.
#
# Usage:
#   .\run_gemv_benchmark.ps1                    # Run benchmark only
#   .\run_gemv_benchmark.ps1 -Profile           # Run with NCU profiling
#   .\run_gemv_benchmark.ps1 -LockClocks        # Lock GPU clocks first
#   .\run_gemv_benchmark.ps1 -M 1 -N 4096       # Custom dimensions
# =====================================================================

param(
    [switch]$Profile,
    [switch]$LockClocks,
    [switch]$Unlock,
    [int]$M = 0,
    [int]$N = 0,
    [int]$Warmup = 0,
    [int]$Iters = 0,
    [string]$Arch = "sm_89"
)

$ErrorActionPreference = "Stop"

# =====================================================================
# Paths
# =====================================================================

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$KernelDir  = Join-Path $ProjectRoot "kernel"
$BenchDir   = Join-Path $ProjectRoot "benchmarks"
$OutputDir  = Join-Path $ProjectRoot "benchmarks\output"

if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# =====================================================================
# Find CUDA Toolkit
# =====================================================================

function Find-CudaHome {
    if ($env:CUDA_HOME) { return $env:CUDA_HOME }
    if ($env:CUDA_PATH) { return $env:CUDA_PATH }
    
    # Probe standard install locations
    $candidates = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    )
    foreach ($c in $candidates) {
        if (Test-Path "$c\bin\nvcc.exe") { return $c }
    }
    throw "CUDA Toolkit not found. Set CUDA_HOME or CUDA_PATH."
}

$CudaHome = Find-CudaHome
$Nvcc = Join-Path $CudaHome "bin\nvcc.exe"
$Ncu  = Join-Path $CudaHome "..\Nsight\ncu.exe"
if (!(Test-Path $Ncu)) {
    # Try alternate location
    $Ncu = "ncu"
}

Write-Host "CUDA Home: $CudaHome" -ForegroundColor Cyan
Write-Host "nvcc:      $Nvcc" -ForegroundColor Cyan

# =====================================================================
# Unlock/Lock GPU Clocks
# =====================================================================

if ($Unlock) {
    Write-Host "`nUnlocking GPU clocks..." -ForegroundColor Yellow
    & nvidia-smi -rgc 2>$null
    Write-Host "Clocks unlocked (default boost behavior)." -ForegroundColor Green
    exit 0
}

if ($LockClocks) {
    Write-Host "`nLocking GPU clocks to maximum..." -ForegroundColor Yellow
    
    # Get max clock from nvidia-smi
    $query = & nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>$null
    if ($query) {
        $maxClock = [int]($query.Trim())
        Write-Host "  Max SM clock: ${maxClock} MHz"
        
        # Lock clocks (requires admin)
        & nvidia-smi -lgc $maxClock,$maxClock
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Clocks locked to ${maxClock} MHz." -ForegroundColor Green
        } else {
            Write-Host "  FAILED: Run PowerShell as Administrator to lock clocks." -ForegroundColor Red
            Write-Host "  Manual command: nvidia-smi -lgc ${maxClock},${maxClock}" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Could not query GPU clocks. Is nvidia-smi in PATH?" -ForegroundColor Red
    }
    Write-Host ""
}

# =====================================================================
# Compile
# =====================================================================

$SourceFile = Join-Path $BenchDir "gemv_benchmark.cu"
$OutputExe  = Join-Path $OutputDir "gemv_benchmark.exe"

Write-Host "`nCompiling benchmark..." -ForegroundColor Yellow

$nvccArgs = @(
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--gpu-architecture=$Arch",
    "-maxrregcount=64",
    "-lineinfo",
    "-I$KernelDir",
    "-o", $OutputExe,
    $SourceFile,
    "-lcublas",
    "-lcudart_static"
)

Write-Host "  nvcc $($nvccArgs -join ' ')" -ForegroundColor DarkGray

& $Nvcc @nvccArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation FAILED." -ForegroundColor Red
    exit 1
}
Write-Host "  Compiled successfully: $OutputExe" -ForegroundColor Green

# =====================================================================
# Run Benchmark
# =====================================================================

Write-Host "`nRunning benchmark..." -ForegroundColor Yellow

$benchArgs = @()
if ($M -gt 0 -and $N -gt 0) {
    $benchArgs += $M.ToString()
    $benchArgs += $N.ToString()
    if ($Warmup -gt 0) { $benchArgs += $Warmup.ToString() }
    if ($Iters -gt 0)  { $benchArgs += $Iters.ToString() }
}

& $OutputExe @benchArgs
$benchExitCode = $LASTEXITCODE

# =====================================================================
# Nsight Compute Profiling (optional)
# =====================================================================

if ($Profile) {
    Write-Host "`n=== Nsight Compute Profiling ===" -ForegroundColor Magenta
    
    $ncuExe = $Ncu
    if (!(Get-Command $ncuExe -ErrorAction SilentlyContinue)) {
        Write-Host "ncu not found in PATH. Attempting CUDA toolkit path..." -ForegroundColor Yellow
        $ncuExe = Join-Path $CudaHome "..\..\..\..\Program Files\NVIDIA Corporation\Nsight Compute 2024.3\ncu.exe"
        if (!(Test-Path $ncuExe)) {
            $ncuExe = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3\ncu.exe"
        }
        if (!(Test-Path $ncuExe)) {
            Write-Host "ncu not found. Install Nsight Compute or add to PATH." -ForegroundColor Red
            Write-Host "Manual profiling commands:" -ForegroundColor Yellow
            Write-Host "  ncu --set full -o custom_profile --kernel-name ternary_zero_gemv_kernel $OutputExe" -ForegroundColor DarkGray
            Write-Host "  ncu --set full -o cublas_profile --kernel-nameFilter 'gemv|hgemv' $OutputExe" -ForegroundColor DarkGray
            exit $benchExitCode
        }
    }
    
    # Profile custom kernel
    Write-Host "`nProfiling custom kernel..." -ForegroundColor Yellow
    $customOut = Join-Path $OutputDir "custom_kernel_profile"
    & $ncuExe --set full `
              --kernel-name "ternary_zero_gemv_kernel" `
              --launch-count 10 `
              --launch-skip 5 `
              -o $customOut `
              $OutputExe
    
    # Profile cuBLAS kernel  
    Write-Host "`nProfiling cuBLAS kernel..." -ForegroundColor Yellow
    $cublasOut = Join-Path $OutputDir "cublas_kernel_profile"
    & $ncuExe --set full `
              --kernel-name-filter "gemv|hgemv" `
              --launch-count 10 `
              --launch-skip 5 `
              -o $cublasOut `
              $OutputExe
    
    Write-Host "`nProfile reports saved:" -ForegroundColor Green
    Write-Host "  Custom:  ${customOut}.ncu-rep" -ForegroundColor Cyan
    Write-Host "  cuBLAS:  ${cublasOut}.ncu-rep" -ForegroundColor Cyan
    Write-Host "`nOpen in Nsight Compute GUI for detailed analysis." -ForegroundColor Cyan
}

# =====================================================================
# Restore Clocks
# =====================================================================

if ($LockClocks) {
    Write-Host "`nRestoring GPU clocks to default..." -ForegroundColor Yellow
    & nvidia-smi -rgc 2>$null
    Write-Host "Clocks restored." -ForegroundColor Green
}

Write-Host "`nDone." -ForegroundColor Green
exit $benchExitCode