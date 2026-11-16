# Packs the Windows release: a zip with everything needed to run
# noiseextractor without installing, and an Inno Setup installer with the
# same files.
#
#   packaging/windows/package.ps1 -Version 1.2.0 -Arch x64|arm64 -OutDir DIR
#
# Expects `cargo build --release -p noiseextractor -p noiseextractor-cli`
# to have run with `-C target-feature=+crt-static` (see
# .github/workflows/release.yml): the programs then contain the C runtime
# themselves and need no Visual C++ redistributable, and as wxWidgets is
# linked in statically, the two .exe files are all there is to ship.
#
# Writes OutDir\noiseextractor-VERSION-windows-ARCH.zip and
# OutDir\noiseextractor-VERSION-windows-ARCH-setup.exe.
param(
    [Parameter(Mandatory)] [string] $Version,
    [Parameter(Mandatory)] [ValidateSet('x64', 'arm64')] [string] $Arch,
    [Parameter(Mandatory)] [string] $OutDir
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root 'target' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = Resolve-Path $OutDir

# --- Folder of the zip release ---------------------------------------------
$name = "noiseextractor-$Version-windows-$Arch"
$temp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$stage = Join-Path $temp $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path $stage | Out-Null

Copy-Item (Join-Path $targetDir 'release\noiseextractor.exe') $stage
Copy-Item (Join-Path $targetDir 'release\noiseextractor-cli.exe') $stage
Copy-Item (Join-Path $root 'packaging\icons\noiseextractor.ico') $stage
Copy-Item (Join-Path $root 'LICENSE') (Join-Path $stage 'LICENSE.txt')
@"
noiseextractor $Version for Windows ($Arch)

noiseextractor.exe       the program
noiseextractor-cli.exe   the command line tool (run "noiseextractor-cli --help")

Third-party software in this folder:

- wxWidgets (built into noiseextractor.exe): https://wxwidgets.org,
  wxWindows Library Licence.

Source code of noiseextractor:
https://github.com/gunterkoenigsmann/noiseextractor
"@ | Set-Content -Encoding utf8 (Join-Path $stage 'README.txt')

# --- Zip ------------------------------------------------------------------------
# The folder itself goes into the zip, so unpacking gives one folder.
$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Get-Item $zip | Format-Table Name, Length

# --- Installer ------------------------------------------------------------------
$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw 'Inno Setup 6 (ISCC.exe) not found' }
& $iscc `
    "/DAppVersion=$Version" "/DArch=$Arch" "/DSourceDir=$stage" "/DOutputDir=$OutDir" `
    (Join-Path $root 'packaging\windows\noiseextractor.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }
Get-Item (Join-Path $OutDir "$name-setup.exe") | Format-Table Name, Length
