# pylint: disable=invalid-name,line-too-long,too-many-locals,too-many-arguments,too-many-branches,too-many-statements,stop-iteration-return
import os
import tqdm
import glob
import string
import zipfile

import numpy as np
import urllib
import hashlib
import urllib.request
import fontTools
import PIL


LIGATURES = {"\U0000FB01": "fi", "\U0000FB02": "fl"}
LIGATURE_STRING = "".join(LIGATURES.keys())

def font_supports_alphabet(filepath, alphabet):
    """Verify that a font contains a specific set of characters.

    Args:
        filepath: Path to fsontfile
        alphabet: A string of characters to check for.
    """
    if alphabet == "":
        return True
    font = fontTools.ttLib.TTFont(filepath)
    if not all(
        any(ord(c) in table.cmap.keys() for table in font["cmap"].tables)
        for c in alphabet
    ):
        return False
    font = PIL.ImageFont.truetype(filepath)
    try:
        for character in alphabet:
            font.getsize(character)
    # pylint: disable=bare-except
    except:
        return False
    return True


def get_fonts(
    cache_dir=None,
    alphabet=string.ascii_letters + string.digits,
    exclude_smallcaps=False,
):
    """Download a set of pre-reviewed fonts.

    Args:
        cache_dir: Where to save the dataset. By default, data will be
            saved to ~/.keras-ocr.
        alphabet: An alphabet which we will use to exclude fonts
            that are missing relevant characters. By default, this is
            set to `string.ascii_letters + string.digits`.
        exclude_smallcaps: If True, fonts that are known to use
            the same glyph for lowercase and uppercase characters
            are excluded.

    Returns:
        A list of font filepaths.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join("~", ".keras-ocr"))
    fonts_zip_path = download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/fonts.zip",
        sha256="d4d90c27a9bc4bf8fff1d2c0a00cfb174c7d5d10f60ed29d5f149ef04d45b700",
        filename="fonts.zip",
        cache_dir=cache_dir,
    )
    fonts_dir = os.path.join(cache_dir, "fonts")
    if len(glob.glob(os.path.join(fonts_dir, "**/*.ttf"))) != 2746:
        print("Unzipping fonts ZIP file.")
        with zipfile.ZipFile(fonts_zip_path) as zfile:
            zfile.extractall(fonts_dir)
    font_filepaths = glob.glob(os.path.join(fonts_dir, "**/*.ttf"))
    if exclude_smallcaps:
        with open(
            download_and_verify(
                url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/fonts_smallcaps.txt",
                sha256="6531c700523c687f02852087530d1ab3c7cc0b59891bbecc77726fbb0aabe68e",
                filename="fonts_smallcaps.txt",
                cache_dir=cache_dir,
            ),
            "r",
            encoding="utf8",
        ) as f:
            smallcaps_fonts = f.read().split("\n")
            smallcaps_fonts = [ origpath.replace('/', os.path.sep) for origpath in smallcaps_fonts ]
            font_filepaths = [
                filepath
                for filepath in font_filepaths
                if os.path.join(*filepath.split(os.sep)[-2:]) not in smallcaps_fonts
            ]
    if alphabet != "":
        font_filepaths = [
            filepath
            for filepath in tqdm.tqdm(font_filepaths, desc="Filtering fonts.")
            if font_supports_alphabet(filepath=filepath, alphabet=alphabet)
        ]
    return font_filepaths


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