from itertools import cycle
import logging
import sys
from threading import Thread
import time

import psutil

# Custom formatter
# Colorize logger output if needed
color2num = dict(
    gray=30,red=31,green=32,
    yellow=33,blue=34,magenta=35,
    cyan=36,white=37,crimson=38,
)

def colorize(
    string: str, 
    color: str, 
    bold: bool = False, 
    highlight: bool = False) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text.

    Args:
        string: The message to colourise
        color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
        bold: If to bold the string
        highlight: If to highlight the string

    Returns:
        Colourised string
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"

class ColoredFormatter(logging.Formatter):

    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.WARNING:
            self._style._fmt = f'{colorize("[%(levelname)s]",color="yellow")} - %(message)s'

        elif record.levelno == logging.INFO:
            self._style._fmt = f'{colorize("[%(levelname)s]",color="green")} - %(message)s'

        elif record.levelno == logging.ERROR:
            self._style._fmt = f'{colorize("[%(levelname)s]",color="crimson",bold=True)} - %(message)s'

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = ColoredFormatter()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

class Loader:
    def __init__(self, bb):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = (
            f"Buffering cell from {bb.LATMIN:.3f}°N-{bb.LATMAX:.3f}°N "
            f"and {bb.LONMIN:.3f}°E-{bb.LONMAX:.3f}°E"
        )
        self.timeout = 0.1

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self.t_start = time.perf_counter()
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"{self.desc} {c}", flush=True, end="\r")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        print(" "*100, end = "\r")
        logger.info(f"{self.desc}")
        self.t_end = time.perf_counter()
        logger.info(f"Cell Buffering completed in [{(self.t_end-self.t_start):.1f} s]")

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()

# Context manager that continuously shows memory usage
# while running the code inside the context.
class MemoryLoader:
    def __init__(self):
        self.timeout = 0.2

        self._thread = Thread(target=self._show_memory_usage, daemon=True)
        self.done = False

    def start(self):
        self.t_start = time.perf_counter()
        self._thread.start()
        return self

    def _show_memory_usage(self):
        """
        Prints memory usage every `timeout` seconds
        """
        while True:
            if self.done:
                break
            print(
                f"Memory usage: {psutil.virtual_memory().percent}% "
                f"[{psutil.virtual_memory().used/1e9:.2f} GB]", 
                end="\r")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        print(" "*100, end = "\r")
        self.t_end = time.perf_counter()
        print(
            f"Loading took {self.t_end - self.t_start:.2f} seconds \n"
            f"Memory usage: {psutil.virtual_memory().percent}% "
            f"[{psutil.virtual_memory().used/1e9:.2f} GB]"
            )

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()
