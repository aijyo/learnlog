# ================================
# Clang + Ninja dev env (VS2022)
# - Import VsDevCmd (MSVC & WinSDK)
# - Prepend Ninja and LLVM/Clang to PATH
# - Set CMake env for Ninja + clang-cl
# - Effective for the CURRENT PS session
# ================================

param(
  # Paths (defaults are based on your message)
  [string]$NinjaHome = 'D:\tools\ninja',
  [string]$LlvmHome  = 'D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64',
  [string]$VsDevCmd  = 'D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat',

  # Use POSIX-style clang/clang++ instead of clang-cl if desired
  [switch]$UseClangPosix
)

# --- Helper: import all environment changes produced by a .bat into this PS process
function Import-BatchEnvironment {
  param(
    [Parameter(Mandatory=$true)][string]$BatchFile,
    [string]$Arguments = ''
  )
  if (-not (Test-Path -LiteralPath $BatchFile)) {
    Write-Warning "VsDevCmd not found: $BatchFile"
    return
  }
  # Run: <batch> <args> && set   and capture the environment snapshot
  $cmd = @('/c', "`"$BatchFile`" $Arguments && set")
  $lines = & cmd.exe $cmd 2>$null
  if (-not $lines) {
    Write-Warning "Failed to import environment from: $BatchFile"
    return
  }
  foreach ($line in $lines) {
    $eq = $line.IndexOf('=')
    if ($eq -gt 0) {
      $name  = $line.Substring(0, $eq)
      $value = $line.Substring($eq + 1)
      # Apply to current process environment
      [System.Environment]::SetEnvironmentVariable($name, $value, 'Process') | Out-Null
    }
  }
  Write-Host "[OK] Imported environment from: $BatchFile"
}

# --- 1) Import MSVC/Windows SDK env from VsDevCmd (x64 host/target)
Import-BatchEnvironment -BatchFile $VsDevCmd -Arguments '-arch=x64 -host_arch=x64'

# --- 2) Prepend Ninja and LLVM bin to PATH (if they exist)
$prepend = @()
if (Test-Path -LiteralPath $NinjaHome) { $prepend += $NinjaHome }
$llvmBin = Join-Path $LlvmHome 'bin'
if (Test-Path -LiteralPath $llvmBin) { $prepend += $llvmBin }

if ($prepend.Count -gt 0) {
  $env:Path = ($prepend -join ';') + ';' + $env:Path
  Write-Host "[OK] PATH updated (prepended): $($prepend -join ';')"
} else {
  Write-Warning "Neither NinjaHome nor LLVM bin path exists. Check your paths."
}

# --- 3) Choose compiler frontend (default: clang-cl on Windows)
if ($UseClangPosix) {
  $env:CC  = 'clang.exe'
  $env:CXX = 'clang++.exe'
  $env:CMAKE_C_COMPILER  = 'clang.exe'
  $env:CMAKE_CXX_COMPILER = 'clang++.exe'
} else {
  $env:CC  = 'clang-cl.exe'
  $env:CXX = 'clang-cl.exe'
  $env:CMAKE_C_COMPILER  = 'clang-cl.exe'
  $env:CMAKE_CXX_COMPILER = 'clang-cl.exe'
}

# --- 4) CMake + Ninja convenience env
$env:CMAKE_GENERATOR     = 'Ninja'
$env:CMAKE_MAKE_PROGRAM  = (Join-Path $NinjaHome 'ninja.exe')
$env:CMAKE_BUILD_PARALLEL_LEVEL = $env:NUMBER_OF_PROCESSORS
# target
$env:TARGET = 'x86_64-pc-windows-msvc'
# --- 5) Show versions (sanity check)
function Show-Tool {
  param([string]$Name, [string[]]$Args = @('--version'))
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) {
    # Use Start-Process to reliably pass arguments and capture outputs
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $cmd.Source
    $psi.Arguments = ($Args -join ' ')
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute        = $false
    $psi.CreateNoWindow         = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEnd() + $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    $first = $out -split "`r?`n" | Where-Object { $_ -match '\S' } | Select-Object -First 1
    Write-Host "[OK] $Name : $first"
  } else {
    Write-Warning "$Name not found in PATH"
  }
}

Show-Tool 'clang-cl'        # If using clang-cl
Show-Tool 'clang'           # If using POSIX driver
Show-Tool 'ninja'
Show-Tool 'cmake'
echo "target is $env:TARGET"

Write-Host ""
Write-Host "=== Dev environment ready ==="
Write-Host "CC=$env:CC"
Write-Host "CXX=$env:CXX"
Write-Host "CMAKE_GENERATOR=$env:CMAKE_GENERATOR"
Write-Host "CMAKE_MAKE_PROGRAM=$env:CMAKE_MAKE_PROGRAM"
Write-Host ""
Write-Host "Examples:"
Write-Host "  cmake -B build_vs -S cpp\lowest --preset works_on_default""
