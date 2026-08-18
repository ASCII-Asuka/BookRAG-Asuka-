param(
    [ValidateSet("source-smoke", "smoke", "full")]
    [string]$Stage = "smoke"
)

$ErrorActionPreference = "Stop"
$Commit = "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9"
$Method = "react_official_repro"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SourceRoot = Join-Path $RepoRoot "runs\third_party\ReAct"
$OutputRoot = Join-Path $RepoRoot "runs\react_official_repro"

if (-not (Test-Path $Python)) {
    throw "Repository Python environment is missing."
}
if (-not (Test-Path (Join-Path $SourceRoot ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path $SourceRoot) | Out-Null
    git clone https://github.com/ysymyth/ReAct.git $SourceRoot
}
git -C $SourceRoot checkout --detach $Commit | Out-Null
$ActualCommit = (git -C $SourceRoot rev-parse HEAD).Trim()
if ($ActualCommit -ne $Commit) {
    throw "Official ReAct source revision mismatch."
}
if ($Stage -eq "source-smoke") {
    Write-Output "Official ReAct source lock verified for $Method."
    exit 0
}

Push-Location $RepoRoot
try {
    & $Python -m Scripts.baselines.react_official.full_experiment `
        --stage $Stage `
        --prepared-root "qasper=runs/hipporag2/full_prepared/qasper" `
        --prepared-root "hotpotqa=runs/hipporag2/full_prepared/hotpotqa" `
        --private-config "qasper=runs/_configs/qasper_full1005_react_0417263dd.yaml" `
        --private-config "hotpotqa=runs/_configs/hotpotqa_react_private.yaml" `
        --public-config "config/react_official_repro.yaml" `
        --source-root $SourceRoot `
        --output-root $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "ReAct reproduction failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

