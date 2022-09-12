###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Various small utility functions
"""
# Used for file hashing utilities
import hashlib
import os
from pathlib import Path
#


def ffs(x):
    """
    Returns the index, counting from 0, of the least significant set bit in `x`.
    """
    return (x & -x).bit_length() - 1


def fls(x):
    """
    Returns the index, counting from 0, of the most significant set bit in `x`.
    """
    return x.bit_length() - 1


def popcount(x):
    """
    Return the number of '1' bits in `x`.
    """
    return bin(x).count('1')


def argmin(values):
    """
    Given an iterable of `values` return the index of the smallest value.
    """
    def argmin_pairs(pairs):
        """
        Given an iterable of `pairs` return the key corresponding to the smallest value
        """
        return min(pairs, key=lambda x: x[1])[0]

    return argmin_pairs(enumerate(values))


def s2u(i):
    """
    Convert signed 8-bit integer `i` to unsigned.
    """
    if i < 0:
        i += 256
    return i


def u2s(i):
    """
    Convert unsigned 8-bit integer `i` to signed.
    """
    if i > 127:
        i -= 256
    return i


def nthone(n, x):
    """
    Return the position of the `n`th 1-bit in `x` (counting starts at bit position 0 to the right).
    Example: n = 2, x = 0xff00 returns 9.
    """
    b = bin(x)
    r = len(b)
    while n > 0:
        r = b.rfind('1', 2, r)
        if r < 0:
            return r
        n -= 1
    return len(b) - r - 1


def overlap(a, b):
    """
    Return true if range `a`[0]/`a`[1] and range `b`[0]/`b`[1] overlap.
    [0] is the start and [1] is the end of the ranges.
    """
    return a[0] >= b[0] and a[0] <= b[1] \
        or a[1] >= b[0] and a[1] <= b[1] \
        or b[0] >= a[0] and b[0] <= a[1] \
        or b[1] >= a[0] and b[1] <= a[1]


def plural(x, name, multiple='s', singular=''):
    """
    Return singular or plural form of variable `name` depending on value `x`.
    """
    if x != 1:  # Works for negative, 0, 2, 3, 4...
        return name + multiple
    return name + singular


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
    alphabetical order.
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

    tmp = file.parent.joinpath("tmp")
    with open(tmp, "w", encoding='utf-8') as f:
        f.write(content)

    match = (hash_file(file) == hash_file(tmp))
    os.remove(tmp)
    return match
