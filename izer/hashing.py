###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
File hashing utility functions used for file content checking
and comparisons
"""

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryFile


def hash_sha1(val):
    """
    Return the SHA1 has of a sequence of bytes
    """
    if not isinstance(val, bytes):
        val = bytes(val, encoding="utf-8")
    return hashlib.sha1(val).digest()


def hash_file(filepath):
    """
    Return the SHA1 hash of a file's contents
    """
    return hash_sha1(open(Path(filepath), 'rb').read())


def hash_folder(folderpath) -> bytes:
    """
    Return the SHA1 hash of a folder's contents.  All files are hashed in
    alphabetical order, but the directories are not.
    """
    folderpath = Path(folderpath)
    result = b''
    for d, _subdirs, files in os.walk(folderpath):
        for f in sorted(files):
            file_path = Path(d).joinpath(f)
            relative_path = file_path.relative_to(folderpath)

            result = hash_sha1(result + hash_file(file_path) + bytes(str(relative_path),
                               encoding="utf-8"))

    return result


def compare_content(content: str, file: Path) -> bool:
    """
    Compare the 'content' string to the existing content in 'file'.

    It seems that when a file gets written there may be some metadata that is affecting
    the hash functions.  As a result, this function writes 'content' to a temporary file,
    then checks for equality using the temp file.
    """
    if not file.exists():
        return False

    tmp = TemporaryFile(mode="w", encoding="utf-8")
    tmp.write(content)

    match = (hash_file(file) == hash_file(tmp))

    tmp.close()
    return match
