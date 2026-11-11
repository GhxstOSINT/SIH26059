$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repoRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $repoRoot 'data\byu-iceberg-v8-manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$targetDir = Join-Path $repoRoot 'work\datasets\byu-iceberg-v8'
$archivePath = Join-Path $targetDir $manifest.archive.file
$partialPath = "$archivePath.partial"
$expectedHash = $manifest.archive.sha256.ToUpperInvariant()
$expectedBytes = [long]$manifest.archive.bytes

New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

if (Test-Path -LiteralPath $archivePath) {
    $existing = Get-Item -LiteralPath $archivePath
    $existingHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    if ($existing.Length -eq $expectedBytes -and $existingHash -eq $expectedHash) {
        Write-Output "Already downloaded and verified: $archivePath"
        exit 0
    }
    throw "An archive already exists at '$archivePath' but does not match the pinned size and SHA-256. It was left untouched."
}

try {
    Invoke-WebRequest -Uri $manifest.download_url -OutFile $partialPath -UseBasicParsing
    $download = Get-Item -LiteralPath $partialPath
    $downloadHash = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash
    if ($download.Length -ne $expectedBytes -or $downloadHash -ne $expectedHash) {
        throw "The downloaded archive did not match the pinned size and SHA-256."
    }
    Move-Item -LiteralPath $partialPath -Destination $archivePath
    Write-Output "Downloaded and verified: $archivePath"
} finally {
    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force
    }
}
