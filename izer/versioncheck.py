###################################################################################################
# Copyright (C) 2021-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Check whether local repo is in sync with upstream
"""
import datetime
import os
import pathlib

import __main__
import git
import github

from .eprint import nprint, wprint

VERSION_CHECK_FILE = ".version-check"


def check_repo(
        upstream: str,
        verbose: bool = False,
) -> bool:
    """
    Check whether the upstream GitHub repository named `upstream` is in sync with the
    local repository's active branch.
    Returns True if the version check succeeded, False otherwise.
    """
    folder = pathlib.Path(__main__.__file__).parent
    if verbose:
        print('Checking for .git in folder', folder)

    if not os.path.isdir(os.path.join(folder, ".git")):
        wprint('Cannot check for updates - no local repository.')
        return False

    try:
        localrepo = git.Repo.init(folder)
        active_branch = localrepo.active_branch
        if verbose:
            print('Active branch:', active_branch, "(dirty)" if localrepo.is_dirty() else "")

        localhead = localrepo.head.commit.hexsha
        localdate = datetime.datetime.utcfromtimestamp(localrepo.head.commit.committed_date)
        if verbose:
            print('LOCAL ', localhead, localdate)
    except ValueError as exc:
        wprint('Cannot check for updates from GitHub -', str(exc))
        return False

    retval = False
    g = github.Github()
    try:
        repo = g.get_repo(upstream)
        branch = repo.get_branch(branch=str(active_branch))
        head = branch.commit.commit.sha
        date = branch.commit.commit.committer.date
        if verbose:
            print('REMOTE', head, date)

        retval = True
        if localhead != head:
            nprint(f'Upstream repository on GitHub has updates on the {active_branch} branch! '
                   '(Use --no-version-check to disable this check.)')
        elif verbose:
            print(f'Local repository is up-to-date on the {active_branch} branch.')

    except github.GithubException as exc:
        wprint(f'Cannot check for updates for git branch "{active_branch}" from GitHub -',
               exc.data['message'])

    return retval


def get_last_check(
    verbose: bool = False,
) -> int:
    """
    Return the date/time of the last version check.
    """
    folder = pathlib.Path(__main__.__file__).parent
    if verbose:
        print('Checking for .version-check in folder', folder)

    version_file = os.path.join(folder, VERSION_CHECK_FILE)

    if not os.path.isfile(version_file):
        if verbose:
            print('No prior version checks (file does not exist)')
        return 0

    with open(version_file, mode='r', encoding='utf-8') as f:
        s = f.readline()

    try:
        retval = int(s)
    except Exception:  # pylint: disable=broad-except
        retval = 0
    return retval


def set_last_check(
    time: int,
    verbose: bool = False,
) -> None:
    """
    Set the last update check to `time`.
    """
    folder = pathlib.Path(__main__.__file__).parent
    if verbose:
        print('Modifying .version-check in folder', folder)

    version_file = os.path.join(folder, VERSION_CHECK_FILE)

    with open(version_file, mode='w', encoding='utf-8') as f:
        f.write(str(time))
