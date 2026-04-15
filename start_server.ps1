$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8000

Write-Host "Serving from: $projectDir"
Write-Host "URL: http://localhost:$port/"

python -m http.server $port --directory "$projectDir"
