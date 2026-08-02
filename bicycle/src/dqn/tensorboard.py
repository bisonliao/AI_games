"""TensorBoard CLI with a bounded port-availability probe.

TensorBoard 2.21's WerkzeugServer checks whether a requested port is occupied
with a blocking ``connect_ex(("localhost", port))``. Some WSL networking modes
silently drop closed-port traffic to 127.0.0.1, so that probe waits for the OS
TCP timeout before TensorBoard binds its server. This module preserves the
normal TensorBoard CLI and plugins, but limits only that startup probe.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import socket
import sys
from typing import Iterator

from absl import app
from tensorboard import default, main_lib, program
from tensorboard.plugins import base_plugin


PORT_PROBE_TIMEOUT_SECONDS = 0.2
_ORIGINAL_SOCKET_CLASS = socket.socket


class _DeadlineSocket(_ORIGINAL_SOCKET_CLASS):
    def connect_ex(self, address):  # type: ignore[no-untyped-def]
        previous_timeout = self.gettimeout()
        self.settimeout(PORT_PROBE_TIMEOUT_SECONDS)
        try:
            return super().connect_ex(address)
        except (TimeoutError, socket.timeout):
            # A timed-out probe means there was no positive evidence that a
            # server owns the port. The subsequent bind remains authoritative
            # and still reports EADDRINUSE without a race-related false start.
            return errno.ETIMEDOUT
        finally:
            self.settimeout(previous_timeout)


@contextmanager
def _bounded_port_probe() -> Iterator[None]:
    original = program.socket.socket
    program.socket.socket = _DeadlineSocket
    try:
        yield
    finally:
        program.socket.socket = original


def create_fast_werkzeug_server(wsgi_app, flags):  # type: ignore[no-untyped-def]
    """Delegate to TensorBoard while bounding its blocking port probe."""
    with _bounded_port_probe():
        return program.create_port_scanning_werkzeug_server(wsgi_app, flags)


def main() -> None:
    """Run the standard TensorBoard CLI with the bounded port-probe server."""
    main_lib.global_init()
    tensorboard = program.TensorBoard(
        plugins=default.get_plugins(),
        server_class=create_fast_werkzeug_server,
    )
    try:
        app.run(tensorboard.main, flags_parser=tensorboard.configure)
    except base_plugin.FlagsError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
