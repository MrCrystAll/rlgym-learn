#!/usr/bin/env python
"""
A Python class implementing KBHIT, the standard keyboard-interrupt poller.
Works transparently on Windows and Posix (Linux, Mac OS X).  Doesn't work
with IDLE.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

import os
import sys

# Windows
if os.name == "nt":
    import io
    import msvcrt

# Posix (Linux, OS X)
else:
    import atexit
    import termios
    from select import select


class KBHit:
    def __init__(self):
        """Creates a KBHit object that you can call to do various keyboard things."""

        if os.name == "nt":
            self.fd = sys.stdin.fileno()
            self._file = io.BufferedReader(
                os.fdopen(self.fd, "rb", buffering=0), buffer_size=1
            )

            self._is_stdin_unblockable = True
            try:
                os.set_blocking(self.fd, False)
            except OSError:
                self._is_stdin_unblockable = False
                self._file.close()

        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = self.new_term[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)

    def set_normal_term(self):
        """Resets to normal terminal.  On Windows this is a no-op."""

        if os.name == "nt":
            self._file.close()
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def getch(self):
        """Returns a keyboard character after kbhit() has been called.
        Should not be called in the same program as getarrow().
        """

        if os.name == "nt":
            # If we are in a situation where stdin is blockable/unblockable, we try to read
            if self._is_stdin_unblockable:
                _stdin_input = self._file.read(1)
                if _stdin_input:
                    return _stdin_input.decode("utf-8")

            # Otherwise the usual
            return msvcrt.getch().decode("utf-8")

        else:
            return sys.stdin.read(1)

    def getarrow(self):
        """Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up
        1 : right
        2 : down
        3 : left
        Should not be called in the same program as getch().
        """

        if os.name == "nt":
            msvcrt.getch()  # skip 0xE0
            c = msvcrt.getch()
            vals = [72, 77, 80, 75]

        else:
            c = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]

        return vals.index(ord(c.decode("utf-8")))

    def kbhit(self):
        """Returns True if keyboard character was hit, False otherwise."""
        if os.name == "nt":
            # If we are in a situation where stdin is blockable/unblockable, we try to peek (non-blocking) to see
            _kbhit = False
            if self._is_stdin_unblockable:
                _kbhit |= len(self._file.peek(1)) == 1
            return _kbhit | msvcrt.kbhit()

        else:
            dr, dw, de = select([sys.stdin], [], [], 0)
            return dr != []


# Test
if __name__ == "__main__":
    kb = KBHit()

    print("Hit any key, or ESC to exit")

    while True:
        if kb.kbhit():
            c = kb.getch()
            if ord(c) == 27:  # ESC
                break
            print(c)

    kb.set_normal_term()
