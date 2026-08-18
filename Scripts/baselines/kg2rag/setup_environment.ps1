param(
    [string]$PythonExecutable = "C:\Users\Asuka\AppData\Local\Programs\Python\Python310\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$SourceDir = Join-Path $RepoRoot "runs\third_party\KG2RAG"
$VenvDir = Join-Path $RepoRoot ".venv-kg2rag"
$Revision = "7d626c77b7af30b55aa3f960cde755b9549a0616"

if (-not (Test-Path -LiteralPath $SourceDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $SourceDir) | Out-Null
    git clone https://github.com/nju-websoft/KG2RAG.git $SourceDir
}

git -C $SourceDir fetch --all --tags
git -C $SourceDir checkout --detach $Revision
$ActualRevision = (git -C $SourceDir rev-parse HEAD).Trim()
if ($ActualRevision -ne $Revision) {
    throw "KG2RAG revision mismatch: expected $Revision, got $ActualRevision"
}

if (-not (Test-Path -LiteralPath $VenvDir)) {
    & $PythonExecutable -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install "numpy>=1.24,<3" "networkx>=2.8,<4" "PyYAML>=6,<7" "requests>=2.31,<3"
& $VenvPython -m pip freeze | Set-Content -Encoding utf8 (Join-Path $VenvDir "pip-freeze.txt")
& $VenvPython --version | Set-Content -Encoding utf8 (Join-Path $VenvDir "python-version.txt")

Write-Host "KG2RAG source and isolated environment are ready."
