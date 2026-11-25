"""The meta stream's wire shape: text fields beside a frame's raw JPEG bytes.

The engine writes `fields["jpeg"] = jpeg_bytes` beside `pts`, `dets` and the modules' fields
(perception/engine/publisher.py:334-352). The product reads the stream with the connection's
decoding off (main.py's `decode_responses=False`), so every field arrives as bytes -- including
the JPEG, whose bytes are not UTF-8.
"""
import acl


class _RawRedis:
    """The response redis-py returns when it does not decode: bytes keys, bytes values."""

    def __init__(self, records):
        self.records = records
        self.reads = 0

    def xrevrange(self, key, count=1):
        return [(b"9-0", {b"v": b"1"})]

    def xread(self, streams, count=1):
        self.reads += 1
        return [(b"iris:meta:cam-1", self.records)]


def test_a_frame_with_a_binary_jpeg_still_yields_its_text_fields():
    fields = {b"v": b"1", b"pts": b"9.000", b"dets": b"[]", b"jpeg": b"\xff\xd8\xff\xe0\x00\x10JFIF",
              b"pose": b'{"shape":[1280,720],"items":[]}'}
    reader = acl.MetaStreamReader(_RawRedis([(b"10-0", fields)]))
    assert reader.read("cam-1") == []                 # primed at the tip: the history is history
    frames = reader.read("cam-1")
    assert [f["id"] for f in frames] == ["10-0"]
    assert frames[0]["fields"]["pts"] == "9.000"
    assert frames[0]["fields"]["pose"] == '{"shape":[1280,720],"items":[]}'
