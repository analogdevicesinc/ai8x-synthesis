###################################################################################################
# Copyright (C) 2022 Maxim Integrated Products Inc. All Rights Reserved.
#
# Maxim Integrated Products Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Console functions
"""
from typing import Iterable, Optional, Sequence, Union

import rich.progress

from . import state

stderr = None


class Progress(rich.progress.Progress):
    """
    Customized progress bar without ETA
    """
    def __init__(self, start: bool = False, **kwargs):
        super().__init__(
            "[progress.description]{task.description}",
            rich.progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            console=stderr,
            redirect_stdout=False,
            redirect_stderr=False,
            disable=not state.display_progress,
            **kwargs,
        )
        if start:
            self.start()


def track(
    sequence: Union[Sequence[rich.progress.ProgressType], Iterable[rich.progress.ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    update_period: float = 0.1,
) -> Iterable[rich.progress.ProgressType]:
    """
    Track progress by iterating over a sequence.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.

    """

    progress = Progress()

    with progress:
        yield from progress.track(
            sequence, total=total, description=description, update_period=update_period
        )
