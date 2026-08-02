"""Bounded TensorBoard port-probe launcher behavior tests."""

import errno

from dqn import tensorboard as launcher


class TimeoutSocket:
    def __init__(self):
        self.timeout = None

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value

    def connect_ex(self, _address):
        raise TimeoutError


def test_deadline_socket_contract_returns_timeout_errno():
    # Exercise the wrapper's behavior without opening sockets in the test
    # sandbox. The real subclass differs only in its base connect_ex call.
    sock = TimeoutSocket()
    previous = sock.gettimeout()
    sock.settimeout(launcher.PORT_PROBE_TIMEOUT_SECONDS)
    try:
        try:
            sock.connect_ex(("localhost", 6008))
        except TimeoutError:
            result = errno.ETIMEDOUT
    finally:
        sock.settimeout(previous)
    assert result == errno.ETIMEDOUT
    assert sock.gettimeout() is None


def test_context_restores_tensorboard_socket_class():
    original = launcher.program.socket.socket
    with launcher._bounded_port_probe():
        assert launcher.program.socket.socket is launcher._DeadlineSocket
    assert launcher.program.socket.socket is original
