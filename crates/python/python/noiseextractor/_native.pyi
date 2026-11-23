# Type stubs of the extension module (crates/python/src/lib.rs). Users
# call the documented wrappers in __init__.py instead.
from typing import List, Optional, Tuple

__version__: str

class Error(Exception): ...

def denoise(
    input: str,
    output: Optional[str],
    reduction_db: float,
    rule: str,
    noise_from: Optional[Tuple[float, float]],
    normalize: Optional[float],
    format: Optional[str],
    bitrate: Optional[int],
    save_project: Optional[str],
    threads: Optional[int],
) -> Optional[dict]: ...
def process(
    input: str,
    output: str,
    project: Optional[str],
    solo: list,
    without: list,
    normalize: Optional[float],
    format: Optional[str],
    bitrate: Optional[int],
    threads: Optional[int],
) -> dict: ...
def mark(
    input: str,
    time: Optional[Tuple[float, float]],
    freq: Optional[Tuple[float, float]],
    outline: Optional[List[Tuple[float, float]]],
    name: str,
    extract: bool,
    project: Optional[str],
    save_project: Optional[str],
) -> dict: ...
def discover(
    input: str,
    sounds: int,
    time: Optional[Tuple[float, float]],
    grouping: bool,
    enable: list,
    extract: bool,
    project: Optional[str],
    save_project: Optional[str],
) -> list: ...
def spectrogram(
    input: str,
    output: str,
    project: Optional[str],
    width: int,
    height: int,
    max_hz: Optional[float],
    log: bool,
    threads: Optional[int],
) -> None: ...
def new_project(path: str) -> None: ...
def denoise_samples(
    channels: list,
    sample_rate: int,
    reduction_db: float,
    rule: str,
    noise_from: Optional[Tuple[float, float]],
    threads: Optional[int],
) -> list: ...
def process_samples(
    channels: list,
    sample_rate: int,
    project: Optional[str],
    solo: list,
    without: list,
    threads: Optional[int],
) -> list: ...
def load(path: str) -> Tuple[list, int]: ...
def run_cli(argv: list) -> int: ...
