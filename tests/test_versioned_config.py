from perception.lib.versioned import VersionedConfig


class _Redis:
    def __init__(self, **kv):
        self.kv, self.writes = dict(kv), []

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value):
        self.writes.append((key, value))
        self.kv[key] = value


def test_a_bump_is_reloaded_then_acknowledged_with_the_version_that_was_applied():
    r = _Redis(v=b"7")
    seen = []
    cfg = VersionedConfig(r, "v", "applied")
    (key, handler), = cfg.control(seen.append).items()
    assert key == "v"
    handler(b"7")
    assert seen == [7] and r.writes == [("applied", 7)]


def test_a_failing_reload_is_not_acknowledged_so_the_app_keeps_warning():
    r = _Redis(v=b"3")

    def boom(_version):
        raise RuntimeError("bad file")
    handler = next(iter(VersionedConfig(r, "v", "applied").control(boom).values()))
    try:
        handler(b"3")
    except RuntimeError:
        pass
    assert r.writes == []


def test_no_version_yet_is_zero_and_the_startup_ack_never_raises():
    cfg = VersionedConfig(_Redis(), "v", "applied")
    assert cfg.version() == 0
    cfg.acknowledge()
    assert cfg.redis.writes == [("applied", 0)]

    class Down:
        def get(self, k):
            raise ConnectionError

    VersionedConfig(Down(), "v", "applied").acknowledge()
