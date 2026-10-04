"""Discover or provision official native CLIs for the desktop, without PATH edits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import Request, urlopen


CLAUDE_DOWNLOADS = "https://downloads.claude.ai/claude-code-releases"
CODEX_RELEASE = "https://api.github.com/repos/openai/codex/releases/latest"


def install_root():
    override = os.getenv("PHGX_CLI_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData/Local")
    elif platform.system() == "Darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local/share")
    return base / "PyHydroGeophysX" / "cli"


def managed_executable(provider):
    suffix = ".exe" if os.name == "nt" else ""
    return install_root() / provider.id / (provider.command + suffix)


def find_executable(provider):
    configured = os.getenv(provider.path_env)
    if configured:
        return shutil.which(configured)  # honour an explicit override
    managed = managed_executable(provider)
    if managed.is_file() and (os.name == "nt" or os.access(managed, os.X_OK)):
        return str(managed)
    executable = shutil.which(provider.command)
    if executable:
        return executable
    candidates = [Path.home() / ".local/bin" / (provider.command + (".exe" if os.name == "nt" else "")),
                  Path.home() / ".local/bin" / provider.command,
                  Path("/opt/homebrew/bin") / provider.command,
                  Path("/usr/local/bin") / provider.command]
    if os.name == "nt":
        local = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData/Local")
        roaming = Path(os.getenv("APPDATA") or Path.home() / "AppData/Roaming")
        candidates += [roaming / "npm" / (provider.command + ".cmd"),
                       Path.home() / ".bun/bin" / (provider.command + ".exe")]
        if provider.id == "codex_cli":
            # Desktop-bundled CLI directories are not always inherited on PATH
            # by apps launched from Explorer / a conda environment.
            try:
                bundled = list((local / "OpenAI/Codex/bin").glob("*/codex.exe"))
                candidates += sorted(bundled, key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                pass
    for candidate in candidates:
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate)
    return None


def find_node(launcher):
    for candidate in (Path(launcher).parent / "node.exe",
                      Path(os.getenv("ProgramFiles") or "C:/Program Files") / "nodejs/node.exe",
                      Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData/Local") / "Programs/nodejs/node.exe"):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("node")


def _read(url):
    request = Request(url, headers={"User-Agent": "PyHydroGeophysX-Studio",
                                   "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return response.read(2 * 1024 * 1024)


def _platform_names():
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch, claude_arch = "aarch64", "arm64"
    elif machine in ("amd64", "x86_64"):
        arch, claude_arch = "x86_64", "x64"
    else:
        raise RuntimeError("Automatic CLI setup requires a 64-bit x64 or ARM64 system.")
    system = platform.system()
    if system == "Windows":
        return f"{arch}-pc-windows-msvc.exe", f"win32-{claude_arch}"
    if system == "Darwin":
        return f"{arch}-apple-darwin", f"darwin-{claude_arch}"
    if system == "Linux":
        return f"{arch}-unknown-linux-musl", f"linux-{claude_arch}"
    raise RuntimeError(f"Automatic CLI setup is not available on {system}.")


def _download(url, path, checksum, progress):
    if not re.fullmatch(r"[a-fA-F0-9]{64}", checksum or ""):
        raise RuntimeError("The official download has no valid SHA-256 checksum.")
    digest, count = hashlib.sha256(), 0
    request = Request(url, headers={"User-Agent": "PyHydroGeophysX-Studio"})
    progress("Downloading the official CLI…")
    with urlopen(request, timeout=30) as response, path.open("wb") as output:
        size = int(response.headers.get("Content-Length") or 0)
        progress("Downloading the official CLI…")
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            count += len(chunk)
            progress(f"Downloading… {count * 100 // size}%" if size
                     else f"Downloading… {count // (1024 * 1024)} MB")
    if digest.hexdigest().lower() != checksum.lower():
        raise RuntimeError("CLI download checksum did not match. Please retry setup.")


def install_cli(provider, progress=lambda text: None):
    """Install only into the app's user directory, validating official checksums."""
    codex_platform, claude_platform = _platform_names()
    target = managed_executable(provider)
    target.parent.mkdir(parents=True, exist_ok=True)
    progress("Finding the official CLI release…")
    if provider.id == "codex_cli":
        release = json.loads(_read(CODEX_RELEASE))
        progress("Preparing the Codex download…")
        asset_name = "codex-" + codex_platform
        if platform.system() != "Windows":
            asset_name += ".tar.gz"
        asset = next((a for a in release["assets"] if a["name"] == asset_name), None)
        if not asset:
            raise RuntimeError("The official Codex release has no download for this system.")
        url, checksum = asset["browser_download_url"], asset.get("digest", "")
        if not url.startswith("https://github.com/openai/codex/releases/download/"):
            raise RuntimeError("Unexpected Codex download source.")
        checksum = checksum.removeprefix("sha256:")
    else:
        version = _read(CLAUDE_DOWNLOADS + "/latest").decode().strip()
        progress("Checking the official Claude Code release…")
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[\w.-]+)?", version):
            raise RuntimeError("The Claude Code download service returned an invalid version.")
        manifest = json.loads(_read(f"{CLAUDE_DOWNLOADS}/{version}/manifest.json"))
        progress("Preparing the Claude Code download…")
        checksum = manifest["platforms"][claude_platform]["checksum"]
        url = f"{CLAUDE_DOWNLOADS}/{version}/{claude_platform}/{target.name}"
    # Nothing is published until the full download, hash and version check pass.
    with tempfile.TemporaryDirectory(prefix="download-", dir=target.parent) as directory:
        download, binary = Path(directory) / "download", Path(directory) / target.name
        _download(url, download, checksum, progress)
        if provider.id == "codex_cli" and platform.system() != "Windows":
            with tarfile.open(download, "r:gz") as archive:
                member = next((m for m in archive.getmembers()
                               if m.isfile() and Path(m.name).name == "codex-" + codex_platform), None)
                if member is None:
                    raise RuntimeError("The Codex archive contains no CLI executable.")
                with archive.extractfile(member) as source, binary.open("wb") as output:
                    shutil.copyfileobj(source, output)
        else:
            download.rename(binary)
        binary.chmod(0o700)
        progress("Checking the CLI installation…")
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=20,
                                env=provider._environment(),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError("The downloaded CLI could not start: " + (result.stderr or result.stdout)[-500:])
        progress("CLI installation complete.")
        binary.replace(target)
    return str(target)


def check_login(provider):
    args = ["login", "status"] if provider.id == "codex_cli" else ["auth", "status"]
    result = subprocess.run(provider._command() + args, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=20,
                            env=provider._environment(),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if provider.id == "codex_cli":
        logged_in = result.returncode == 0 and "logged in using chatgpt" in (result.stdout + result.stderr).lower()
    else:
        try:
            status = json.loads(result.stdout)
            logged_in = result.returncode == 0 and status.get("authMethod") in ("claude.ai", "oauth_token")
        except (ValueError, AttributeError):
            logged_in = False
    return {"logged_in": logged_in,
            "message": "Signed in. Ready to chat." if logged_in else "Setup complete. Click Log In to continue."}


def prepare_cli(provider, progress=lambda text: None):
    progress("Looking for your CLI…")
    try:
        provider._command()
    except FileNotFoundError:
        if os.getenv(provider.path_env):
            raise RuntimeError("The configured CLI cannot start. Check its custom path.")
        install_cli(provider, progress)
        provider._command()
    progress("Checking login…")
    return check_login(provider)
