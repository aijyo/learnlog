<# 
.SYNOPSIS
  Detect dynamic input dimensions in ONNX models, force-freeze them to static shapes,
  and optionally simplify with onnxsim (robust fallbacks included).

.DESCRIPTION
  - Step 1: Analyze inputs and print which dims are dynamic (dim_param/unknown).
  - Step 2: Force-freeze dynamic dims to integers (Batch/S/others) by editing the model in Python.
  - Step 3: Try onnxsim with --no-large-tensor (and fallbacks); if still failing, keep the frozen model.

  This script is PowerShell 5 compatible (no ternary operator used).
  It treats "output file exists" as success, because some onnxsim builds print the summary to stderr
  while still generating the output successfully.

.PARAMETER Models
  ONNX files to process. Defaults to enc.onnx, df_dec.onnx, erb_dec.onnx in current directory.

.PARAMETER S
  Integer to replace the dynamic dim named by -SName. Default: 120.

.PARAMETER SName
  Name of the time/sequence dynamic dimension to freeze. Default: "S".

.PARAMETER Batch
  If the first dimension is dynamic/batch-like, set it to this value. Default: 1.

.PARAMETER ReplaceOtherDynamicsWith
  Value for any other dynamic dims not matching SName/batch-like. Default: 1.

.PARAMETER InstallDeps
  Install/upgrade Python packages: onnx, onnxruntime, onnxsim.

.PARAMETER Python
  Python launcher (default "python").

.PARAMETER SkipSim
  If present, skip onnxsim; just output the frozen model.

.PARAMETER DryRun
  Only analyze & print intended shapes; no files are written.

.EXAMPLES
  .\Freeze-OnnxDynamicDim.ps1 -S 1
  .\Freeze-OnnxDynamicDim.ps1 -Models ".\enc.onnx",".\df_dec.onnx",".\erb_dec.onnx" -S 160 -InstallDeps
  .\Freeze-OnnxDynamicDim.ps1 -S 120 -SkipSim   # only freeze shapes, do not simplify
#>

[CmdletBinding()]
param(
  [string[]]$Models = @("enc.onnx","df_dec.onnx","erb_dec.onnx"),
  [int]$S = 120,
  [string]$SName = "S",
  [int]$Batch = 1,
  [int]$ReplaceOtherDynamicsWith = 1,
  [switch]$InstallDeps,
  [string]$Python = "python",
  [switch]$SkipSim,
  [switch]$DryRun
)

function Write-Info($msg)  { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err ($msg)  { Write-Host "[ERR ] $msg" -ForegroundColor Red }

# ----- Python helper: analyzer (print inputs and build overwrite shapes) -----
function New-AnalyzerPy {
  $code = @'
import sys, json
import onnx

# argv: model_path, batch, S, SName, otherDyn
model_path = sys.argv[1]
batch = int(sys.argv[2])
S = int(sys.argv[3])
SName = sys.argv[4]
otherDyn = int(sys.argv[5])

def is_batch_like(name: str) -> bool:
    if not name: return False
    name_l = name.lower()
    return name_l in ("b","batch","n","batch_size")

m = onnx.load(model_path)

report = {"model": model_path, "inputs": [], "overwrite_shapes": []}

for inp in m.graph.input:
    ttype = inp.type.tensor_type
    dims_desc = []
    dims_num  = []
    if ttype.HasField("shape"):
        for i, d in enumerate(ttype.shape.dim):
            if d.HasField("dim_value"):
                dims_desc.append(str(d.dim_value))
                dims_num.append(d.dim_value)
            elif d.HasField("dim_param"):
                name = d.dim_param
                dims_desc.append(f"{name}*")
                if i == 0 and is_batch_like(name):
                    dims_num.append(batch)
                elif name == SName:
                    dims_num.append(S)
                elif i == 0 and not name:
                    dims_num.append(batch)
                else:
                    dims_num.append(otherDyn)
            else:
                dims_desc.append("?*")
                dims_num.append(batch if i == 0 else otherDyn)
    else:
        dims_desc.append("<unknown>")
        dims_num.append(otherDyn)

    report["inputs"].append({"name": inp.name, "desc": dims_desc})
    report["overwrite_shapes"].append(f"{inp.name}:{','.join(str(v) for v in dims_num)}")

print(json.dumps(report, ensure_ascii=False))
'@
  $tmp = New-TemporaryFile
  $py  = [System.IO.Path]::ChangeExtension($tmp.FullName, ".py")
  Move-Item $tmp.FullName $py -Force
  Set-Content -LiteralPath $py -Value $code -NoNewline -Encoding UTF8
  return $py
}

# ----- Python helper: freezer (force set static dims across inputs/outputs/value_info, recurse subgraphs) -----
function New-FreezerPy {
  $code = @'
import sys, onnx
from onnx import helper, shape_inference

# argv: in_path, out_path, batch, S, SName, otherDyn
in_path  = sys.argv[1]
out_path = sys.argv[2]
batch    = int(sys.argv[3])
S        = int(sys.argv[4])
SName    = sys.argv[5]
otherDyn = int(sys.argv[6])

def is_batch_like(name: str) -> bool:
    if not name: return False
    name_l = name.lower()
    return name_l in ("b","batch","n","batch_size")

def fix_shape(shape):
    if not shape: return
    for i, d in enumerate(shape.dim):
        if d.HasField("dim_value"):
            continue
        if d.HasField("dim_param"):
            nm = d.dim_param
            if i == 0 and is_batch_like(nm):
                d.ClearField("dim_param"); d.dim_value = batch
            elif nm == SName:
                d.ClearField("dim_param"); d.dim_value = S
            elif i == 0 and not nm:
                d.ClearField("dim_param"); d.dim_value = batch
            else:
                d.ClearField("dim_param"); d.dim_value = otherDyn
        else:
            # Unknown dimension: fill by convention
            d.dim_value = batch if i == 0 else otherDyn

def process_graph(g):
    # Fix input/output/value_info shapes
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            fix_shape(tt.shape)

    # Recurse into subgraphs if any
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                process_graph(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for sg in a.graphs:
                    process_graph(sg)

m = onnx.load(in_path)
process_graph(m.graph)

# Try shape inference (non-strict) to propagate shapes after freezing
try:
    m = shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=False)
except Exception as e:
    # If inference fails, still proceed with the frozen shapes
    pass

onnx.save(m, out_path)
'@
  $tmp = New-TemporaryFile
  $py  = [System.IO.Path]::ChangeExtension($tmp.FullName, ".py")
  Move-Item $tmp.FullName $py -Force
  Set-Content -LiteralPath $py -Value $code -NoNewline -Encoding UTF8
  return $py
}

function Ensure-Dependencies {
  if ($InstallDeps) {
    Write-Info "Installing/Upgrading Python dependencies: onnx, onnxruntime, onnxsim"
    & $Python -m pip install -U onnx onnxruntime onnxsim
    if ($LASTEXITCODE -ne 0) {
      Write-Err "pip install failed."
      throw "pip install failed"
    }
  } else {
    Write-Info "Skipping pip install. Make sure onnx / onnxruntime / onnxsim are available."
  }
}

function Analyze-Model {
  param([string]$ModelPath,[string]$AnalyzerPy)
  $args = @($AnalyzerPy, $ModelPath, $Batch, $S, $SName, $ReplaceOtherDynamicsWith)
  $json = & $Python $args
  if ($LASTEXITCODE -ne 0) { Write-Err "Analyzer failed: $ModelPath"; throw "Analyzer failed" }
  try { return $json | ConvertFrom-Json } catch { Write-Err "Bad analyzer JSON"; Write-Host $json; throw }
}

function Freeze-Model {
  param([string]$ModelPath,[string]$OutFrozen,[string]$FreezerPy)
  $args = @($FreezerPy, $ModelPath, $OutFrozen, $Batch, $S, $SName, $ReplaceOtherDynamicsWith)
  & $Python $args
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutFrozen)) {
    Write-Err "Freezer failed: $ModelPath"
    throw "Freezer failed"
  }
}

function Try-OnnxSim {
  param(
    [string]$ModelPath,   # input (frozen)
    [string]$OutPath      # output (final)
  )

  # Attempt A: onnxsim with --no-large-tensor (no input-shape needed because model is already static)
  $cmdA = @("-m","onnxsim",$ModelPath,$OutPath,"--no-large-tensor")
  Write-Info "Running: $Python $($cmdA -join ' ')"
  & $Python $cmdA
  $code = $LASTEXITCODE
  if ($code -eq 0 -or (Test-Path -LiteralPath $OutPath)) { return 0 }

  Write-Warn "onnxsim A failed (exit=$code). Trying fallback B: --skip-optimization ..."
  # Attempt B: onnxsim with --skip-optimization
  $cmdB = @("-m","onnxsim",$ModelPath,$OutPath,"--no-large-tensor","--skip-optimization")
  Write-Info "Running: $Python $($cmdB -join ' ')"
  & $Python $cmdB
  $code = $LASTEXITCODE
  if ($code -eq 0 -or (Test-Path -LiteralPath $OutPath)) { return 0 }

  Write-Warn "onnxsim B failed (exit=$code). Will keep the frozen model without sim."
  return 1
}

# -------------------- main --------------------
$AnalyzerPy = $null
$FreezerPy  = $null
try {
  Ensure-Dependencies

  $AnalyzerPy = New-AnalyzerPy
  $FreezerPy  = New-FreezerPy
  Write-Info "Analyzer: $AnalyzerPy"
  Write-Info "Freezer : $FreezerPy"

  foreach ($m in $Models) {
    $rp = $null
    try { $rp = Resolve-Path -LiteralPath $m -ErrorAction Stop } catch { Write-Warn "Model not found: $m (skip)"; continue }
    $modelPath = $rp.Path

    Write-Host ""
    Write-Host "====== Analyzing: $modelPath ======" -ForegroundColor Green
    $rep = Analyze-Model -ModelPath $modelPath -AnalyzerPy $AnalyzerPy

    foreach ($inp in $rep.inputs) {
      $hasDyn = $false
      foreach ($d in $inp.desc) {
        if ($d -is [string]) { if ($d.Contains('*')) { $hasDyn = $true; break } }
      }
      $tag = "          "
      if ($hasDyn) { $tag = "[DYNAMIC]" }
      Write-Host ("  {0} Input {1}  Shape: [{2}]" -f $tag, $inp.name, ($inp.desc -join ", "))
    }

    $dir  = Split-Path -Parent $modelPath
    $name = [System.IO.Path]::GetFileNameWithoutExtension($modelPath)
    $frozen = Join-Path $dir ($name + "_frozen_tmp.onnx")
    $final  = Join-Path $dir ($name + "_fixed.onnx")

    Write-Info "Freeze shapes -> $frozen"
    if (-not $DryRun) {
      Freeze-Model -ModelPath $modelPath -OutFrozen $frozen -FreezerPy $FreezerPy
    } else {
      Write-Warn "DryRun enabled. Skipping freeze/simplify for $modelPath"
      continue
    }

    if ($SkipSim) {
      Write-Warn "SkipSim enabled. Keeping frozen model as final: $final"
      Copy-Item -LiteralPath $frozen -Destination $final -Force
      Write-Host "==> Wrote: $final" -ForegroundColor Green
      Remove-Item $frozen -Force -ErrorAction SilentlyContinue
      continue
    }

    $rc = Try-OnnxSim -ModelPath $frozen -OutPath $final
    if ($rc -ne 0) {
      # Last fallback: keep the frozen model
      Copy-Item -LiteralPath $frozen -Destination $final -Force
      Write-Warn "Kept frozen model without onnxsim: $final"
    } else {
      Write-Host "==> Wrote: $final" -ForegroundColor Green
    }
    Remove-Item $frozen -Force -ErrorAction SilentlyContinue
  }

} finally {
  if ($AnalyzerPy -and (Test-Path $AnalyzerPy)) { Remove-Item $AnalyzerPy -Force -ErrorAction SilentlyContinue }
  if ($FreezerPy  -and (Test-Path $FreezerPy )) { Remove-Item $FreezerPy  -Force -ErrorAction SilentlyContinue }
}
