; Inno Setup script for the Windows installer of noiseextractor.
;
; Compiled by packaging/windows/package.ps1, which passes:
;   /DAppVersion=1.2.0          full version (also used in the file name)
;   /DArch=x64 or /DArch=arm64  the architecture the binaries are built for
;   /DSourceDir=...             folder with the files of the zip release
;   /DOutputDir=...             where the installer goes
;
; Needs Inno Setup 6.3 or newer (the "x64compatible" and "arm64"
; architecture identifiers).

#ifndef AppVersion
  #error AppVersion must be defined (/DAppVersion=...)
#endif
#ifndef Arch
  #error Arch must be defined (/DArch=x64 or /DArch=arm64)
#endif

[Setup]
; Stays the same across versions, so that a new version replaces the old one
; instead of being installed next to it. Never change it.
AppId={{C044BB6D-E683-4341-99A3-D55E0CA29079}
AppName=noiseextractor
AppVersion={#AppVersion}
AppVerName=noiseextractor {#AppVersion}
AppPublisher=noiseextractor contributors
AppPublisherURL=https://github.com/gunterkoenigsmann/noiseextractor
AppSupportURL=https://github.com/gunterkoenigsmann/noiseextractor/issues
DefaultDirName={autopf}\noiseextractor
DefaultGroupName=noiseextractor
DisableProgramGroupPage=yes
LicenseFile={#SourceDir}\LICENSE.txt
SetupIconFile={#SourceDir}\noiseextractor.ico
UninstallDisplayIcon={app}\noiseextractor.ico
OutputDir={#OutputDir}
OutputBaseFilename=noiseextractor-{#AppVersion}-windows-{#Arch}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-machine by default, but users without admin rights may choose to
; install for themselves only.
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
; 64-bit only. An x64 build also runs on Windows on Arm (emulated), but the
; arm64 build only there.
#if Arch == "arm64"
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
#else
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#endif
; Windows 10 or newer.
MinVersion=10.0
; Tell Explorer that the "Open with" entries below changed.
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "projects"; Description: "Open noiseextractor projects (.nxp) with noiseextractor"; GroupDescription: "File associations:"
Name: "openwith"; Description: "Offer noiseextractor in ""Open with"" for audio files (WAV, FLAC, Ogg, MP3, M4A, AAC)"; GroupDescription: "File associations:"

[Files]
; Everything of the zip release: both programs, the icon, the license.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; "Open with" entries for audio files. They never make noiseextractor the
; default program: OpenWithProgids only adds it to the list. HKA is
; HKLM for an all-users installation and HKCU for a per-user one.
; The extensions must equal those of noiseextractor_core::audio::FILE_TYPES;
; crates/core/tests/packaging.rs checks it.
Root: HKA; Subkey: "Software\Classes\noiseextractor.audio"; ValueType: string; ValueName: ""; ValueData: "Audio recording"; Flags: uninsdeletekey; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\noiseextractor.audio\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\noiseextractor.ico"; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\noiseextractor.audio\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\noiseextractor.exe"" ""%1"""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "noiseextractor"; Flags: uninsdeletekey; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.wav\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.flac\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.ogg\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.oga\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.mp3\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.m4a\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.aac\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".wav"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".flac"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".ogg"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".oga"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".mp3"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".m4a"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".aac"; ValueData: ""; Tasks: openwith

; Projects (noiseextractor_core::project_file::PROJECT_FILE_TYPE): the
; extension is ours, so noiseextractor becomes its default program (the
; default value of the .nxp key, removed again on uninstall). Must match the
; extension in project_file.rs; crates/core/tests/packaging.rs checks it.
Root: HKA; Subkey: "Software\Classes\noiseextractor.project"; ValueType: string; ValueName: ""; ValueData: "noiseextractor project"; Flags: uninsdeletekey; Tasks: projects
Root: HKA; Subkey: "Software\Classes\noiseextractor.project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\noiseextractor.ico"; Tasks: projects
Root: HKA; Subkey: "Software\Classes\noiseextractor.project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\noiseextractor.exe"" ""%1"""; Tasks: projects
Root: HKA; Subkey: "Software\Classes\.nxp"; ValueType: string; ValueName: ""; ValueData: "noiseextractor.project"; Flags: uninsdeletevalue; Tasks: projects
Root: HKA; Subkey: "Software\Classes\.nxp"; ValueType: string; ValueName: "Content Type"; ValueData: "application/x-noiseextractor-project"; Flags: uninsdeletevalue; Tasks: projects
Root: HKA; Subkey: "Software\Classes\.nxp\OpenWithProgids"; ValueType: string; ValueName: "noiseextractor.project"; ValueData: ""; Flags: uninsdeletevalue; Tasks: projects
Root: HKA; Subkey: "Software\Classes\Applications\noiseextractor.exe\SupportedTypes"; ValueType: string; ValueName: ".nxp"; ValueData: ""; Flags: uninsdeletevalue; Tasks: projects

[Icons]
Name: "{autoprograms}\noiseextractor"; Filename: "{app}\noiseextractor.exe"; IconFilename: "{app}\noiseextractor.ico"
Name: "{autodesktop}\noiseextractor"; Filename: "{app}\noiseextractor.exe"; IconFilename: "{app}\noiseextractor.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\noiseextractor.exe"; Description: "{cm:LaunchProgram,noiseextractor}"; Flags: nowait postinstall skipifsilent
