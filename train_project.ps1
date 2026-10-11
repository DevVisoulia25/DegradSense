param([int]$Epochs = 20, [string]$RunName = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($RunName) {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -m rul train --epochs $Epochs --output "artifacts/$RunName"
} else {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -m rul train --epochs $Epochs
}
if ($LASTEXITCODE -ne 0) { throw 'Training failed. See the error above.' }
