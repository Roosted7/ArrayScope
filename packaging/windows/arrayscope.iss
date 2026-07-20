; Inno Setup script for the ArrayScope Windows installer.
;
; Compiled by packaging/windows/build_installer.ps1 (or CI) after PyInstaller
; has produced build\pyinstaller\dist\ArrayScope. Pass the version in:
;
;   ISCC.exe /DAppVersion=0.8.0 packaging\windows\arrayscope.iss
;
; The installer is the conventional wizard Windows users expect: per-user by
; default (no admin prompt; "install for all users" available via dialog),
; Start Menu entry, optional desktop icon, optional file associations,
; clean uninstaller. Python is invisible — the PyInstaller bundle carries it.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define RepoRoot SourcePath + "\..\.."
#define BundleDir RepoRoot + "\build\pyinstaller\dist\ArrayScope"

[Setup]
AppId={{9D3C1A76-2B84-4F0E-A6E3-52D10FF83A4B}
AppName=ArrayScope
AppVersion={#AppVersion}
AppVerName=ArrayScope {#AppVersion}
AppPublisher=ArrayScope contributors
AppPublisherURL=https://github.com/Roosted7/ArrayScope
AppSupportURL=https://github.com/Roosted7/ArrayScope/issues
AppUpdatesURL=https://github.com/Roosted7/ArrayScope/releases
DefaultDirName={autopf}\ArrayScope
DefaultGroupName=ArrayScope
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#RepoRoot}\dist
OutputBaseFilename=ArrayScope-Setup-{#AppVersion}
SetupIconFile={#RepoRoot}\arrayscope\resources\icons\arrayscope.ico
UninstallDisplayIcon={app}\ArrayScope.exe
LicenseFile={#RepoRoot}\LICENSE
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
ChangesAssociations=yes
DisableProgramGroupPage=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate"; Description: "Associate ArrayScope with its own data formats (.npy, .npz, .cfl, .rec, .nii, .mat)"; GroupDescription: "File associations:"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\ArrayScope"; Filename: "{app}\ArrayScope.exe"
Name: "{autodesktop}\ArrayScope"; Filename: "{app}\ArrayScope.exe"; Tasks: desktopicon

[Registry]
; ProgIDs. Owned formats (ArrayScope defines the type; no widely deployed
; owner exists) take the extension default under the "associate" task. Shared
; formats (DICOM, HDF5) only join the "Open with" list — mirrors
; arrayscope/desktop/filetypes.py on the desktop-integration branch.
; --- owned: .npy ---
Root: HKA; Subkey: "Software\Classes\ArrayScope.npy"; ValueType: string; ValueData: "NumPy array"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.npy\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.npy\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.npy"; ValueType: string; ValueData: "ArrayScope.npy"; Tasks: associate; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.npy\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.npy"; ValueData: ""; Flags: uninsdeletevalue
; --- owned: .npz ---
Root: HKA; Subkey: "Software\Classes\ArrayScope.npz"; ValueType: string; ValueData: "NumPy array archive"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.npz\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.npz\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.npz"; ValueType: string; ValueData: "ArrayScope.npz"; Tasks: associate; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.npz\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.npz"; ValueData: ""; Flags: uninsdeletevalue
; --- owned: .cfl ---
Root: HKA; Subkey: "Software\Classes\ArrayScope.cfl"; ValueType: string; ValueData: "BART complex-float array"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.cfl\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.cfl\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.cfl"; ValueType: string; ValueData: "ArrayScope.cfl"; Tasks: associate; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.cfl\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.cfl"; ValueData: ""; Flags: uninsdeletevalue
; --- owned: .rec ---
Root: HKA; Subkey: "Software\Classes\ArrayScope.rec"; ValueType: string; ValueData: "Philips XML/REC image"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.rec\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.rec\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.rec"; ValueType: string; ValueData: "ArrayScope.rec"; Tasks: associate; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.rec\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.rec"; ValueData: ""; Flags: uninsdeletevalue
; --- owned: .nii (.nii.gz cannot be registered as a compound extension) ---
Root: HKA; Subkey: "Software\Classes\ArrayScope.nii"; ValueType: string; ValueData: "NIfTI neuroimaging data"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.nii\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.nii\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.nii"; ValueType: string; ValueData: "ArrayScope.nii"; Tasks: associate; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.nii\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.nii"; ValueData: ""; Flags: uninsdeletevalue
; --- owned: .mat ---
Root: HKA; Subkey: "Software\Classes\ArrayScope.mat"; ValueType: string; ValueData: "MATLAB data file"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.mat\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.mat\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.mat"; ValueType: string; ValueData: "ArrayScope.mat"; Tasks: associate; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.mat\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.mat"; ValueData: ""; Flags: uninsdeletevalue
; --- shared: .dcm, .h5, .hdf5 ("Open with" only, never the default) ---
Root: HKA; Subkey: "Software\Classes\ArrayScope.dcm"; ValueType: string; ValueData: "DICOM image"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.dcm\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.dcm\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.dcm\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.dcm"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\ArrayScope.h5"; ValueType: string; ValueData: "HDF5 data file"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\ArrayScope.h5\DefaultIcon"; ValueType: string; ValueData: "{app}\ArrayScope.exe,0"
Root: HKA; Subkey: "Software\Classes\ArrayScope.h5\shell\open\command"; ValueType: string; ValueData: """{app}\ArrayScope.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.h5\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.h5"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.hdf5\OpenWithProgids"; ValueType: string; ValueName: "ArrayScope.h5"; ValueData: ""; Flags: uninsdeletevalue

[Run]
Filename: "{app}\ArrayScope.exe"; Description: "{cm:LaunchProgram,ArrayScope}"; Flags: nowait postinstall skipifsilent
