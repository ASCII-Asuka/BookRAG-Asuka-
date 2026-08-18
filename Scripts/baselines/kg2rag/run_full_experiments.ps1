param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$MethodConfig = "config\kg2rag_official.yaml",
    [string]$QasperConfig = "Scripts\cfg\Qasper.yaml",
    [string]$HotpotConfig = "Scripts\cfg\HotpotQA.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RunRoot = Join-Path $RepoRoot "runs\kg2rag"
$StatusPath = Join-Path $RunRoot "full_experiment_status.json"
$LockPath = Join-Path $RunRoot "full_experiment.pid"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Write-ExperimentStatus {
    param([string]$Status, [string]$Stage, [string]$ErrorMessage = "")
    @{
        status = $Status
        stage = $Stage
        updated_at = [DateTimeOffset]::Now.ToString("o")
        pid = $PID
        error = $ErrorMessage
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & (Join-Path $RepoRoot $Python) @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed with exit code $LASTEXITCODE"
    }
}

if (Test-Path -LiteralPath $LockPath) {
    $ExistingPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $LockPath -Raw).Trim(), [ref]$ExistingPid)
    if ($ExistingPid -gt 0 -and (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue)) {
        throw "A KG2RAG experiment process is already active."
    }
}
Set-Content -LiteralPath $LockPath -Value $PID -Encoding ascii

try {
    $Datasets = @(
        @{Name = "qasper"; Config = $QasperConfig},
        @{Name = "hotpotqa"; Config = $HotpotConfig}
    )
    foreach ($Dataset in $Datasets) {
        $PreparedRoot = Join-Path $RunRoot "$($Dataset.Name)\prepared"
        $RetrievalRoot = Join-Path $RunRoot "$($Dataset.Name)\retrieval"

        Write-ExperimentStatus -Status "running" -Stage "$($Dataset.Name)_prepare"
        Invoke-CheckedPython -Arguments @(
            "-m", "Scripts.baselines.kg2rag.prepare",
            "--dataset-config", $Dataset.Config,
            "--output-root", $PreparedRoot
        )

        Write-ExperimentStatus -Status "running" -Stage "$($Dataset.Name)_index_retrieve"
        Invoke-CheckedPython -Arguments @(
            "-m", "Scripts.baselines.kg2rag.run",
            "--prepared-root", $PreparedRoot,
            "--config", $MethodConfig,
            "--output-root", $RetrievalRoot,
            "--stage", "all"
        )

        Write-ExperimentStatus -Status "running" -Stage "$($Dataset.Name)_finalize"
        Invoke-CheckedPython -Arguments @(
            "-m", "Scripts.baselines.kg2rag.finalize",
            "--retrieval-manifest", (Join-Path $RetrievalRoot "manifest.json"),
            "--config", $MethodConfig
        )

        Write-ExperimentStatus -Status "running" -Stage "$($Dataset.Name)_evaluate"
        Invoke-CheckedPython -Arguments @(
            "Eval\evaluation.py",
            "-d", $Dataset.Config,
            "--method", "kg2rag_official",
            "--max_workers", "1"
        )
    }
    Write-ExperimentStatus -Status "complete" -Stage "complete"
}
catch {
    Write-ExperimentStatus -Status "failed" -Stage "failed" -ErrorMessage $_.Exception.Message
    throw
}
finally {
    if (Test-Path -LiteralPath $LockPath) {
        Remove-Item -LiteralPath $LockPath -Force
    }
}
