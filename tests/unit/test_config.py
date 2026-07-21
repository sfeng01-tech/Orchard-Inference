from orchard_inference.config import Settings


def test_default_server_port_is_5000() -> None:
    assert Settings().port == 5000
