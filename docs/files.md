# Project files, file associations and shell completion

## Project files (`*.nxp`)

The GUI's *Save project* and the CLI's `--save-project` / `new-project` write
the filter list as JSON. Projects have their own extension, `.nxp`, so that
the desktop can recognise them: the packages register it as a file type, a
double click opens the project in noiseextractor, and shells complete it.

- A name typed into the GUI's save dialog without an extension gets `.nxp`.
  The CLI writes exactly the name it is given.
- The content is still plain JSON (a text editor can open it), and older
  projects saved as `*.json` or `*.nxp.json` keep loading: the open dialog
  offers them, and the GUI recognises a project by its first character
  (`{`) when the extension is not one it knows.
- The type is defined in one place, `PROJECT_FILE_TYPE` in
  `crates/core/src/project_file.rs`: extension `nxp`, MIME type
  `application/x-noiseextractor-project` (a subclass of `application/json`),
  macOS UTI `io.github.gunterkoenigsmann.noiseextractor.project`.

## Starting the GUI with files

`noiseextractor [RECORDING] [PROJECT]` opens a recording, a project, or both,
in any order. File managers pass every selected file (`Exec=noiseextractor
%F`), so selecting a recording and its project and choosing "Open with
noiseextractor" opens them together. Further files are reported in the log
and ignored. On macOS, files from Finder (double click, "Open With", a drop on
the Dock icon) arrive as an event rather than as arguments and are opened
the same way.

## File associations per package

| Package | Audio files (WAV, FLAC, Ogg, MP3, M4A, AAC) | Projects (`.nxp`) |
| --- | --- | --- |
| `.deb` | "Open with" (desktop file `MimeType`) | Default program; type declared in `/usr/share/mime/packages/noiseextractor.xml`, icon `application-x-noiseextractor-project` |
| AppImage | Through an integration tool (appimaged, AppImageLauncher, Gear Lever), which installs the desktop file | The same tools install the MIME declaration from the AppImage's `usr/share/mime/packages` |
| Windows installer | "Open with" (task *Offer noiseextractor in "Open with"*) | Default program (task *Open noiseextractor projects*, on by default) |
| macOS DMG | "Open With" (`LSHandlerRank` Alternate) | Owner of the exported type: double click opens it |
| Snap | "Open with" (snap desktop file) | Not registered: snapd does not install MIME types from snaps. Open projects with File > Open project, or pass them on the command line |

None of the packages makes noiseextractor the default program for audio
files. `crates/core/tests/packaging.rs` checks that the desktop files,
`noiseextractor.xml`, `Info.plist` and the Inno Setup script agree with
`FILE_TYPES` and `PROJECT_FILE_TYPE`.

A program built with `cargo` installs nothing. To get the project type and
"Open with" from a source build on Linux, copy the files yourself:

```sh
install -Dm644 packaging/linux/noiseextractor.xml ~/.local/share/mime/packages/noiseextractor.xml
install -Dm644 packaging/linux/noiseextractor.desktop ~/.local/share/applications/noiseextractor.desktop
install -Dm644 packaging/icons/noiseextractor.svg ~/.local/share/icons/hicolor/scalable/apps/noiseextractor.svg
install -Dm644 packaging/icons/noiseextractor.svg \
    ~/.local/share/icons/hicolor/scalable/mimetypes/application-x-noiseextractor-project.svg
update-mime-database ~/.local/share/mime
update-desktop-database ~/.local/share/applications
```

(`Exec=noiseextractor` must then be on your `PATH`, e.g. via `cargo install
--path crates/gui`.)

## Shell completion

`noiseextractor-cli completions SHELL [PROGRAM]` prints a completion script.
For `noiseextractor-cli` (the default) it completes subcommands and options
and is generated from the CLI's own definition (bash, zsh, fish, PowerShell,
Elvish). For `noiseextractor` (the GUI) it completes directories, the audio
types above and `.nxp` projects (bash, zsh, fish).

The `.deb` installs the bash, zsh and fish scripts of both programs. With
the other packages, or a source build, install them yourself, e.g.

```sh
# bash
mkdir -p ~/.local/share/bash-completion/completions
noiseextractor-cli completions bash > ~/.local/share/bash-completion/completions/noiseextractor-cli
noiseextractor-cli completions bash noiseextractor > ~/.local/share/bash-completion/completions/noiseextractor
# zsh (a directory on your $fpath)
noiseextractor-cli completions zsh > ~/.zfunc/_noiseextractor-cli
noiseextractor-cli completions zsh noiseextractor > ~/.zfunc/_noiseextractor
# fish
noiseextractor-cli completions fish > ~/.config/fish/completions/noiseextractor-cli.fish
noiseextractor-cli completions fish noiseextractor > ~/.config/fish/completions/noiseextractor.fish
# PowerShell (add to $PROFILE)
noiseextractor-cli completions powershell | Out-String | Invoke-Expression
```

On macOS the CLI is `noiseextractor.app/Contents/MacOS/noiseextractor-cli`;
with the AppImage, use `./noiseextractor-*.AppImage --cli completions ...`
(and a `noiseextractor-cli` symlink to the AppImage on your `PATH`). The
snap's commands are `noiseextractor` and `noiseextractor.cli`; its scripts
are not installed yet.
