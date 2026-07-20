# Build the ArrayScope Windows installer and portable zip.
#
# Prerequisites (see packaging/README.md):
#   - Python environment with the repo installed:  pip install ".[installer]"
#   - Inno Setup 6:  winget install JRSoftware.InnoSetup  (or choco install innosetup)
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build_installer.ps1
#
# Outputs:
#   dist\ArrayScope-Setup-<version>.exe            (wizard installer)
#   dist\ArrayScope-<version>-windows-x86_64-portable.zip

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Version = python -c "from arrayscope._version import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0) { throw "Could not read version (is the repo installed in this environment?)" }
Write-Host "==> PyInstaller bundle (ArrayScope $Version)"

pyinstaller --noconfirm --distpath build\pyinstaller\dist --workpath build\pyinstaller\work `
    packaging\pyinstaller\arrayscope.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Locate ISCC.exe: PATH first, then the default install locations.
$Iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($Iscc) {
    $Iscc = $Iscc.Source
} else {
    $Candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:LocalAppData}\Programs\Inno Setup 6\ISCC.exe"
    )
    $Iscc = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Iscc) { throw "Inno Setup 6 not found; install with: winget install JRSoftware.InnoSetup" }
}

Write-Host "==> Compiling installer ($Iscc)"
& $Iscc /DAppVersion=$Version packaging\windows\arrayscope.iss
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }

Write-Host "==> Portable zip"
New-Item -ItemType Directory -Force -Path dist | Out-Null
$Zip = "dist\ArrayScope-$Version-windows-x86_64-portable.zip"
if (Test-Path $Zip) { Remove-Item $Zip }
Compress-Archive -Path build\pyinstaller\dist\ArrayScope -DestinationPath $Zip

Write-Host "==> Done:"
Write-Host "    dist\ArrayScope-Setup-$Version.exe"
Write-Host "    $Zip"
