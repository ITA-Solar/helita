"""
tools to deal with I/O on the shell
"""
import sys


def progressbar(i, m):
    ''' Draws a progress bar with = signs and percentage.
        IN: i (step of the loop), m (max of the loop).'''
    sys.stdout.write('\r')
    sys.stdout.write("[%-30s] %d%%" %
                     ('=' * int(round(i * 30. / m)), 100. * i / m))
    sys.stdout.flush()
    if i == m:
        sys.stdout.write('\n')


class Getch:
    """
    Multi-platform getch
    Gets a single character from standard input.
    Does not echo to the screen.
    """
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty
        import sys

    def __call__(self):
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()
