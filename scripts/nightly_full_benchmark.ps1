param(
    [string]$Model = "gemma4:e2b",
    [int]$SummaryLimit = 200,
    [string]$QuestionsPath = "data/eval_questions_alt.json",
    [int]$SmokeQuestions = 3,
    [int]$JudgeWorkers = 2,
    [int]$BenchmarkRetries = 3,
    [string]$RunStamp = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$graphPath = Join-Path $repoRoot "data\graphs\knowledge_graph.graphml"
$runStamp = if ($RunStamp) { $RunStamp } else { Get-Date -Format "yyyyMMdd_HHmmss" }
$runDir = Join-Path $repoRoot "results\overnight_runs\$runStamp"
$logPath = Join-Path $runDir "overnight_benchmark.log"
$auditPath = Join-Path $runDir "session_audit.md"

$smokeCsv = Join-Path $runDir "benchmark_smoke.csv"
$smokeMd = Join-Path $runDir "benchmark_smoke.md"
$smokeJsonl = Join-Path $runDir "benchmark_smoke.jsonl"
$fullCsv = Join-Path $runDir "benchmark_full.csv"
$fullMd = Join-Path $runDir "benchmark_full.md"
$fullJsonl = Join-Path $runDir "benchmark_full.jsonl"

$startTime = Get-Date
$runStatus = "RUNNING"
$failureMessage = ""

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Write-Stage {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ""
    Write-Host "[$timestamp] $Message"
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Stage "START $Name"
    & $Action
    Write-Stage "DONE  $Name"
}

function Invoke-External {
    param(
        [string]$Description,
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-ArtifactLine {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path
        return "- $Path ($([Math]::Round($item.Length / 1KB, 1)) KB)"
    }

    return "- $Path (missing)"
}

function Invoke-BenchmarkWithResume {
    param(
        [string]$Name,
        [string]$CsvPath,
        [string]$MarkdownPath,
        [string]$JsonlPath,
        [int]$MaxQuestions = 0
    )

    for ($attempt = 1; $attempt -le $BenchmarkRetries; $attempt++) {
        try {
            $arguments = @(
                "-m", "src.cli", "benchmark", $QuestionsPath,
                "--llm-model", $Model,
                "--output", $CsvPath,
                "--markdown-output", $MarkdownPath,
                "--resume-jsonl", $JsonlPath,
                "--resume",
                "--judge-workers", "$JudgeWorkers",
                "--faithfulness",
                "--quality-judges"
            )
            if ($MaxQuestions -gt 0) {
                $arguments += @("--max-questions", "$MaxQuestions")
            }

            Invoke-External -Description $Name -FilePath $python -Arguments $arguments
            return
        }
        catch {
            if ($attempt -ge $BenchmarkRetries) {
                throw
            }
            $delay = [Math]::Min(60, 10 * $attempt)
            Write-Stage "RETRY $Name attempt $attempt failed: $($_.Exception.Message). Sleeping ${delay}s before resume."
            Start-Sleep -Seconds $delay
        }
    }
}

Push-Location $repoRoot
Start-Transcript -Path $logPath -Force | Out-Null

try {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Virtual environment Python not found: $python"
    }

    Invoke-Step "Refresh community sidecar" {
        Invoke-External -Description "Refresh community sidecar" -FilePath $python -Arguments @(
            "-m", "src.cli", "refresh-communities",
            "--graph-path", $graphPath,
            "--summary-limit", "$SummaryLimit",
            "--llm-model", $Model
        )
    }

    Invoke-Step "Benchmark preflight" {
        Invoke-External -Description "Benchmark preflight" -FilePath $python -Arguments @(
            "-m", "src.cli", "benchmark", $QuestionsPath, "--llm-model", $Model, "--preflight-only"
        )
    }

    Invoke-Step "Smoke benchmark" {
        Invoke-BenchmarkWithResume `
            -Name "Smoke benchmark" `
            -CsvPath $smokeCsv `
            -MarkdownPath $smokeMd `
            -JsonlPath $smokeJsonl `
            -MaxQuestions $SmokeQuestions
    }

    foreach ($artifact in @($smokeCsv, $smokeMd, $smokeJsonl)) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            throw "Smoke benchmark did not produce expected artifact: $artifact"
        }
    }

    Invoke-Step "Full benchmark" {
        Invoke-BenchmarkWithResume `
            -Name "Full benchmark" `
            -CsvPath $fullCsv `
            -MarkdownPath $fullMd `
            -JsonlPath $fullJsonl
    }

    foreach ($artifact in @($fullCsv, $fullMd, $fullJsonl)) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            throw "Full benchmark did not produce expected artifact: $artifact"
        }
    }

    $runStatus = "SUCCESS"
}
catch {
    $runStatus = "FAILED"
    $failureMessage = $_.Exception.Message
    Write-Stage "FAIL  $failureMessage"
}
finally {
    $endTime = Get-Date
    $duration = New-TimeSpan -Start $startTime -End $endTime
    $gitStatus = git status --short
    $gitDiffStat = git diff --stat
    $failureLine = if ($failureMessage) { "- $failureMessage" } else { "- None" }

    @(
        "# Overnight Benchmark Audit"
        ""
        "- Status: $runStatus"
        "- Started: $($startTime.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
        "- Finished: $($endTime.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
        "- Duration: $([string]$duration)"
        "- Model: $Model"
        "- Question set: $QuestionsPath"
        "- Community summary limit: $SummaryLimit"
        "- Smoke question count: $SmokeQuestions"
        "- Judge workers: $JudgeWorkers"
        "- Benchmark retries: $BenchmarkRetries"
        ""
        "## Artifacts"
        (Get-ArtifactLine -Path $logPath)
        "- $auditPath (this file)"
        (Get-ArtifactLine -Path $smokeCsv)
        (Get-ArtifactLine -Path $smokeMd)
        (Get-ArtifactLine -Path $smokeJsonl)
        (Get-ArtifactLine -Path $fullCsv)
        (Get-ArtifactLine -Path $fullMd)
        (Get-ArtifactLine -Path $fullJsonl)
        ""
        "## Failure"
        $failureLine
        ""
        "## Git Status"
        '```text'
        ($gitStatus -join "`n")
        '```'
        ""
        "## Git Diff Stat"
        '```text'
        ($gitDiffStat -join "`n")
        '```'
    ) | Set-Content -Path $auditPath -Encoding UTF8

    Stop-Transcript | Out-Null
    Pop-Location

    if ($runStatus -ne "SUCCESS") {
        exit 1
    }
}
