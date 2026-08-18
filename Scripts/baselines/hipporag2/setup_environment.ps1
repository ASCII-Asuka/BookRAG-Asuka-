param(
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$sourceRoot = Join-Path $repoRoot "runs\third_party"
$sourcePath = Join-Path $sourceRoot "HippoRAG"
$archivePath = Join-Path $sourceRoot "HippoRAG-c617143f.zip"
$environmentPath = Join-Path $repoRoot ".venv-hipporag2"
$recordPath = Join-Path $repoRoot "runs\hipporag2_environment"
$commit = "c617143f01477243992a63b2e2151cc003dd3b21"
$archiveUrl = "https://github.com/OSU-NLP-Group/HippoRAG/archive/$commit.zip"
$archiveHash = "ef5831ca2f00d1367849054002936b2775b42f384742baa96578994681685a96"

New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
if (-not (Test-Path (Join-Path $sourcePath "setup.py"))) {
    & git clone https://github.com/OSU-NLP-Group/HippoRAG.git $sourcePath
    $cloneSucceeded = $LASTEXITCODE -eq 0
    if ($cloneSucceeded) {
        & git -C $sourcePath checkout $commit
        if ($LASTEXITCODE -ne 0) {
            throw "HippoRAG cloned successfully but checkout of $commit failed."
        }
    }
    else {
        if (Test-Path $sourcePath) {
            throw "Git left a partial source directory at $sourcePath. Move it aside, then rerun the script."
        }
        Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath -UseBasicParsing
        $actualHash = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLowerInvariant()
        if ($actualHash -ne $archiveHash) {
            throw "HippoRAG archive SHA256 mismatch: expected $archiveHash, got $actualHash"
        }
        Expand-Archive -LiteralPath $archivePath -DestinationPath $sourceRoot -Force
        Move-Item -LiteralPath (Join-Path $sourceRoot "HippoRAG-$commit") -Destination $sourcePath
    }
}

if (Test-Path (Join-Path $sourcePath ".git")) {
    $actualCommit = (git -C $sourcePath rev-parse HEAD).Trim()
    if ($actualCommit -ne $commit) {
        throw "HippoRAG commit mismatch: expected $commit, got $actualCommit"
    }
}
elseif (Test-Path $archivePath) {
    $actualHash = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLowerInvariant()
    if ($actualHash -ne $archiveHash) {
        throw "HippoRAG archive SHA256 mismatch: expected $archiveHash, got $actualHash"
    }
}

if (-not (Test-Path $environmentPath)) {
    py -3.10 -m venv $environmentPath
}
$python = Join-Path $environmentPath "Scripts\python.exe"

function Invoke-CheckedPython([string[]]$CommandArgs) {
    & $python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($CommandArgs -join ' ')"
    }
}

Invoke-CheckedPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--progress-bar", "off")
Invoke-CheckedPython @("-m", "pip", "install", "-e", $sourcePath, "--no-deps", "--no-build-isolation", "--progress-bar", "off")

if (-not $SkipDependencies) {
    $packages = @(
        "torch==2.5.1",
        "transformers==4.45.2",
        "openai>=1.91.0",
        "networkx==3.4.2",
        "python-igraph==0.11.8",
        "tiktoken==0.7.0",
        "pydantic==2.10.4",
        "tenacity==8.5.0",
        "einops",
        "tqdm",
        "boto3",
        "nest_asyncio",
        "numpy",
        "pandas",
        "pyarrow",
        "requests",
        "scipy",
        "filelock",
        "httpx",
        "packaging",
        "sentence-transformers==3.1.1",
        "aiohttp",
        "click",
        "importlib-metadata",
        "jinja2",
        "jsonschema",
        "python-dotenv"
    )
    Invoke-CheckedPython (@("-m", "pip", "install") + $packages + @("--progress-bar", "off"))
    # Both modules are imported eagerly by the official package, although this
    # adapter does not execute their local-model or benchmark paths.
    Invoke-CheckedPython @("-m", "pip", "install", "litellm==1.73.1", "gritlm==1.0.2", "--no-deps", "--progress-bar", "off")
}

New-Item -ItemType Directory -Force -Path $recordPath | Out-Null
Invoke-CheckedPython @(
    "-c",
    "import importlib.metadata as m; import importlib.util; import hipporag; from hipporag import Chunk, HippoRAG; assert m.version('hipporag') == '2.0.0a4'; assert importlib.util.find_spec('vllm') is None; c=Chunk(content='smoke', source_id='p1', metadata={'paragraph_id': 1}); assert c.source_id == 'p1' and c.metadata['paragraph_id'] == 1"
)
& $python --version 2>&1 | Set-Content -Path (Join-Path $recordPath "python-version.txt") -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw "Unable to record Python version" }
& $python -m pip freeze | Set-Content -Path (Join-Path $recordPath "pip-freeze.txt") -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw "Unable to record pip freeze" }
$commit | Set-Content -Path (Join-Path $recordPath "source-revision.txt") -Encoding UTF8
Write-Host "HippoRAG 2 environment ready: $environmentPath"
