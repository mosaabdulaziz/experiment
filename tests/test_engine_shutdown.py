"""The engine stops cleanly on SIGTERM (`docker stop`, the safety controller stopping a role): the main loop quits so
its cleanup runs and deletes this process's iris:status keys -- otherwise a stopped sidecar's streams stay listed as
running forever (the keys have no expiry). perception/engine/main.py only imports inside the DeepStream image, so this
reads the source."""
import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parents[1] / "perception" / "engine" / "main.py").read_text()


def test_sigterm_quits_the_main_loop_before_the_cleanup():
    assert re.search(r"GLib\.unix_signal_add\([^)]*signal\.SIGTERM", SRC)
    handler_at = SRC.index("signal.SIGTERM")
    assert handler_at < SRC.index("loop.run()")
    assert "loop.quit()" in SRC[SRC.rfind("def ", 0, handler_at):handler_at]
