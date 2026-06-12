param(
    [string]$Stage = "smoke",
    [string]$CombinedLog = "C:\tmp\matlab_logs\physionet_stage.log",
    [string]$PrefDir = "C:\tmp\matlab_prefdir",
    [string]$SubjectList = "",
    [string]$CaseList = "",
    [switch]$ForceRerun
)

$workspaceRoot = "C:\Users\yuges\OneDrive\Desktop\Project1\PhysioNet_8Case_Benchmark"
$frameworkDir = Join-Path $workspaceRoot "benchmark_framework"
$matlabExe = "C:\Program Files\MATLAB\R2026a\bin\matlab.exe"

New-Item -ItemType Directory -Force -Path "C:\tmp" | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CombinedLog) | Out-Null
New-Item -ItemType Directory -Force -Path $PrefDir | Out-Null

$env:TEMP = "C:\tmp"
$env:TMP = "C:\tmp"
$env:MATLAB_PREFDIR = $PrefDir
$env:BENCH_STAGE_NAME = $Stage
if ([string]::IsNullOrWhiteSpace($SubjectList)) {
    Remove-Item Env:BENCH_SUBJECT_LIST -ErrorAction SilentlyContinue
} else {
    $env:BENCH_SUBJECT_LIST = $SubjectList
}
if ([string]::IsNullOrWhiteSpace($CaseList)) {
    Remove-Item Env:BENCH_CASE_LIST -ErrorAction SilentlyContinue
} else {
    $env:BENCH_CASE_LIST = $CaseList
}
if ($ForceRerun) {
    $env:BENCH_FORCE_RERUN = "1"
} else {
    Remove-Item Env:BENCH_FORCE_RERUN -ErrorAction SilentlyContinue
}

& $matlabExe -sd $frameworkDir -batch run_stage_from_env *>> $CombinedLog
