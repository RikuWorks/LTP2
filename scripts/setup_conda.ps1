param(
    [string]$EnvName = "lite-therm-pose",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $RootDir "environment.yml"

function Write-Log {
    param([string]$Message)
    Write-Host "[LiteThermPose] $Message"
}

function Fail {
    param([string]$Message)
    throw "[LiteThermPose][ERROR] $Message"
}

function Find-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    Fail "conda.exe が見つかりません。Anaconda または Miniconda をインストールしてから再実行してください。"
}

$CondaExe = Find-Conda

if (!(Test-Path $EnvFile)) {
    Fail "environment.yml が見つかりません: $EnvFile"
}

Write-Log "creating or updating conda environment: $EnvName"
& $CondaExe env remove -n $EnvName -y *> $null

if ($CpuOnly) {
    & $CondaExe create -n $EnvName -y `
        "python=3.10" `
        "pip" `
        "numpy>=1.24" `
        "opencv>=4.8" `
        "pillow>=10" `
        "pyyaml>=6" `
        "tqdm>=4.66" `
        "pytorch>=2.2" `
        "torchvision>=0.17" `
        "cpuonly" `
        -c pytorch -c conda-forge
    & $CondaExe run -n $EnvName pip install -e $RootDir
} else {
    & $CondaExe env create -n $EnvName -f $EnvFile
}

Write-Log "verifying torch CUDA availability"
& $CondaExe run -n $EnvName python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

Write-Log "activation:"
Write-Host "  conda activate $EnvName"
Write-Host "  python scripts/verify_gpu.py"
