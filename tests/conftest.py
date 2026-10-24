"""Every test runs offline: sockets to the network raise, and no QPU arming leaks in."""

import os
import socket

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

_QPU_ENV = ("DQ_QPU", "DQ_QPU_TASK", "DQ_QPU_MAX_JOBS", "DQ_QPU_MAX_SHOTS", "DQ_QPU_LEDGER", "IQM_TOKEN")


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(what):
        raise NetworkBlocked(f"tests must not reach the network ({what})")

    def connect(sock, address):
        if sock.family == socket.AF_UNIX:
            return real_connect(sock, address)
        refuse(f"connect {address!r}")

    def connect_ex(sock, address):
        if sock.family == socket.AF_UNIX:
            return real_connect_ex(sock, address)
        refuse(f"connect_ex {address!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: refuse(f"create_connection {address!r}"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: refuse(f"getaddrinfo {host!r}"))


@pytest.fixture(autouse=True)
def unarmed(monkeypatch, tmp_path):
    for name in _QPU_ENV:
        monkeypatch.delenv(name, raising=False)
    # never the real ledger: a path that does not exist unless a test creates it
    monkeypatch.setenv("DQ_QPU_LEDGER", str(tmp_path / "no-ledger.md"))
