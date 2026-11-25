import pytest

from netguard import is_local_client


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "10.1.2.3", "192.168.0.9", "172.20.0.4", "169.254.3.3", "fd00::5"])
def test_local_and_private_addresses_are_allowed(host):
    assert is_local_client(host)


@pytest.mark.parametrize("host", ["8.8.8.8", "2001:4860::8888", "testclient", "", None])
def test_public_or_unknown_clients_are_refused(host):
    assert not is_local_client(host)
