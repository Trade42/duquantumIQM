import socket
from pathlib import Path


def test_published_winners_replay_without_network_or_credentials(monkeypatch):
    from examples.replay_winners import REPOSITORY, main
    monkeypatch.chdir(REPOSITORY)
    original_open = Path.open

    def no_credentials(path, *args, **kwargs):
        if path.name in {"apikey.jom", ".env"}:
            raise AssertionError("Offline replay tried to read credentials.")
        return original_open(path, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("Offline replay tried to use the network.")

    monkeypatch.setattr(Path, "open", no_credentials)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    result = main(["--json"])
    assert result["replayed_shots"] == 36000
    assert not result["hardware_access"]
    assert result["policies"]["X"]["family"] == "mwpm"
    assert result["policies"]["Z"]["family"] == "posterior_10"
    assert result["policies"]["X"]["failures"] == [159, 404, 728]
    assert result["policies"]["Z"]["failures"] == [145, 639, 1147]
    assert socket.create_connection is no_network
