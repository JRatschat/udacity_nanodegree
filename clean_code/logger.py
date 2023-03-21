"""
Logger functionality.

Author: Jonathan Ratschat
Date: 21.03.2022
"""

import logging

from constants import log_path


def load_logger(file_pth: str):
    logging.basicConfig(
        filename=log_path / file_pth,
        level=logging.INFO,
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )

    return logging
