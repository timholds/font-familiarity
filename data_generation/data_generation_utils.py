# pylint: disable=invalid-name,line-too-long,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,stop-iteration-return
import os

import glob

import zipfile

import numpy as np
import urllib
import hashlib
import urllib.request


LIGATURES = {"\U0000FB01": "fi", "\U0000FB02": "fl"}
LIGATURE_STRING = "".join(LIGATURES.keys())


def download_and_verify(url, sha256=None, cache_dir=None, verbose=True, filename=None):
    """Download a file to a cache directory and verify it with a sha256
    hash.

    Args:
        url: The file to download
        sha256: The sha256 hash to check. If the file already exists and the hash
            matches, we don't download it again.
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.
        verbose: Whether to log progress
        filename: The filename to use for the file. By default, the filename is
            derived from the URL.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if filename is None:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    filepath = os.path.join(cache_dir, filename)
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    if verbose:
        print("Looking for " + filepath)
    if not os.path.isfile(filepath) or (sha256 and sha256sum(filepath) != sha256):
        if verbose:
            print("Downloading " + filepath)
        urllib.request.urlretrieve(url, filepath)
    assert sha256 is None or sha256 == sha256sum(
        filepath
    ), "Error occurred verifying sha256."
    return filepath

def get_backgrounds(cache_dir=None):
    """Download a set of pre-reviewed backgrounds.

    Args:
        cache_dir: Where to save the dataset. By default, data will be
            saved to ~/.keras-ocr.

    Returns:
        A list of background filepaths.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join("~", ".keras-ocr"))
    backgrounds_dir = os.path.join(cache_dir, "backgrounds")
    backgrounds_zip_path = download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/backgrounds.zip",
        sha256="f263ed0d55de303185cc0f93e9fcb0b13104d68ed71af7aaaa8e8c91389db471",
        filename="backgrounds.zip",
        cache_dir=cache_dir,
    )
    if len(glob.glob(os.path.join(backgrounds_dir, "*"))) != 1035:
        with zipfile.ZipFile(backgrounds_zip_path) as zfile:
            zfile.extractall(backgrounds_dir)
    return glob.glob(os.path.join(backgrounds_dir, "*.jpg"))

def sha256sum(filename):
    """Compute the sha256 hash for a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):  # type: ignore
            h.update(mv[:n])
    return h.hexdigest()


def get_default_cache_dir():
    return os.environ.get(
        "KERAS_OCR_CACHE_DIR", os.path.expanduser(os.path.join("~", ".keras-ocr"))
    )


def download_and_verify(url, sha256=None, cache_dir=None, verbose=True, filename=None):
    """Download a file to a cache directory and verify it with a sha256
    hash.

    Args:
        url: The file to download
        sha256: The sha256 hash to check. If the file already exists and the hash
            matches, we don't download it again.
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.
        verbose: Whether to log progress
        filename: The filename to use for the file. By default, the filename is
            derived from the URL.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if filename is None:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    filepath = os.path.join(cache_dir, filename)
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    if verbose:
        print("Looking for " + filepath)
    if not os.path.isfile(filepath) or (sha256 and sha256sum(filepath) != sha256):
        if verbose:
            print("Downloading " + filepath)
        urllib.request.urlretrieve(url, filepath)
    assert sha256 is None or sha256 == sha256sum(
        filepath
    ), "Error occurred verifying sha256."
    return filepath


if __name__ == "__main__":
    # Example usage
    get_backgrounds()