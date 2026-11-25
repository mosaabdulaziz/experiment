import base64
import json
import struct

from acl import AppearanceStreamReader, MetaStreamReader, decode_embedding


class FakeRedis:
    """xrevrange/xread over in-memory streams: {key: [(id, fields)]}."""

    def __init__(self, streams):
        self.streams = streams

    def xrevrange(self, key, count=1):
        return list(reversed(self.streams.get(key, [])))[:count]

    def xread(self, cursors, count=None, block=None):
        out = []
        for key, last in cursors.items():
            lo = tuple(int(p) for p in last.split("-"))
            rows = [(rid, f) for rid, f in self.streams.get(key, [])
                    if tuple(int(p) for p in rid.decode().split("-")) > lo][:count]
            if rows:
                out.append((key.encode(), rows))
        return out


def meta(ms, boxes):
    return (f"{ms}-0".encode(), {b"dets": b"[]", b"person_boxes": json.dumps(boxes).encode()})


def emb16(values):
    return base64.b64encode(struct.pack(f"<{len(values)}e", *values)).decode()


def test_read_frames_primes_then_returns_one_frame_per_record():
    r = FakeRedis({"iris:meta:barber-c": [meta(1000, [])]})
    reader = MetaStreamReader(r)
    assert reader.read_frames("barber-c") == []            # primes at the tip
    r.streams["iris:meta:barber-c"] += [meta(1100, [[10, 20, 30, 40, 7]]), meta(1200, [])]
    frames = reader.read_frames("barber-c")
    assert [f.ts for f in frames] == [1.1, 1.2]
    assert frames[0].camera == "barber-c" and frames[0].tracks[0].track_id == 7
    assert frames[0].tracks[0].box == (10, 20, 30, 40)
    assert frames[1].tracks == ()
    assert reader.read_frames("barber-c") == []


def test_appearance_reader_decodes_items():
    items = [{"track_id": 7, "reid": emb16([3.0, 4.0]), "staff_p": 0.2}, {"track_id": 8, "staff_p": 0.9}]
    r = FakeRedis({"iris:appearance:barber-l": [(b"5-0", {b"items": b"[]"})]})
    reader = AppearanceStreamReader(r)
    assert reader.read("barber-l") == []
    r.streams["iris:appearance:barber-l"].append((b"2500-0", {b"items": json.dumps(items).encode()}))
    apps = reader.read("barber-l")
    assert [(a.track_id, a.ts, a.staff_p) for a in apps] == [(7, 2.5, 0.2), (8, 2.5, 0.9)]
    assert apps[0].reid == (0.6, 0.8) and apps[1].reid is None


def test_decode_embedding_normalizes():
    v = decode_embedding(emb16([0.0, 2.0]))
    assert v == (0.0, 1.0)


def test_face_candidate_reader_decodes_items():
    from acl import FaceCandidateReader
    item = {"track_id": 7, "crop_path": "faces/2026-09-25/a.jpg", "embedding": emb16([3.0, 4.0]),
            "quality": 0.61, "face_box": [1, 2, 40, 40]}
    r = FakeRedis({"iris:customer_face:barber-c": [(b"5-0", {b"item": b"{}"})]})
    reader = FaceCandidateReader(r)
    assert reader.read("barber-c") == []
    r.streams["iris:customer_face:barber-c"].append((b"2500-0", {b"item": json.dumps(item).encode()}))
    (c,) = reader.read("barber-c")
    assert (c.camera, c.track_id, c.ts, c.crop_path, c.quality) == ("barber-c", 7, 2.5, "faces/2026-09-25/a.jpg", 0.61)
    assert c.embedding == (0.6, 0.8)
    assert c.face_box == (1.0, 2.0, 40.0, 40.0)


def test_face_reader_without_a_face_box_still_reads():
    # an older sidecar (or a malformed box): the engine falls back to the person's box
    from acl import FaceCandidateReader
    r = FakeRedis({"iris:customer_face:ph-1": [(b"5-0", {b"item": b"{}"})]})
    reader = FaceCandidateReader(r)
    reader.read("ph-1")
    for i, extra in enumerate(({}, {"face_box": None}, {"face_box": [1, 2, 3]}, {"face_box": "x"},
                               {"face_box": [1, 2, 0, 5]})):
        item = {"track_id": 7, "crop_path": "faces/a.jpg", "embedding": emb16([3.0, 4.0]), **extra}
        r.streams["iris:customer_face:ph-1"].append((f"{2500 + i}-0".encode(), {b"item": json.dumps(item).encode()}))
    assert [c.face_box for c in reader.read("ph-1")] == [None] * 5


def test_appearance_reader_carries_the_body_crop():
    items = [{"track_id": 7, "staff_p": 0.2, "crop_path": "bodies/2026-09-25/b.jpg", "crop_quality": 0.5}]
    r = FakeRedis({"iris:appearance:barber-l": [(b"5-0", {b"items": b"[]"})]})
    reader = AppearanceStreamReader(r)
    reader.read("barber-l")
    r.streams["iris:appearance:barber-l"].append((b"2600-0", {b"items": json.dumps(items).encode()}))
    (a,) = reader.read("barber-l")
    assert (a.crop_path, a.crop_quality) == ("bodies/2026-09-25/b.jpg", 0.5)


from acl import tracks_from


def test_tracks_from_boxes_only():
    (t,) = tracks_from(json.dumps([[10, 20, 30, 40, 7]]), None)
    assert (t.track_id, t.box, t.conf, t.outline) == (7, (10, 20, 30, 40), None, None)


def test_tracks_from_merges_masks_by_id():
    boxes = json.dumps([[10, 20, 30, 40, 7], [1, 2, 3, 4, 8]])
    masks = json.dumps([[7, 0.91, [0.1, 0.1, 0.2, 0.1, 0.2, 0.2]], [8, 0.4, None]])
    a, b = tracks_from(boxes, masks)
    assert a.conf == 0.91 and a.outline == (0.1, 0.1, 0.2, 0.1, 0.2, 0.2)
    assert b.conf == 0.4 and b.outline is None


def test_bad_person_masks_are_dropped_not_raised():
    boxes = json.dumps([[10, 20, 30, 40, 7], [1, 2, 3, 4, 8], [5, 5, 5, 5, 9]])
    masks = json.dumps([[7, 0.9, [0.1, 0.1, 0.2]],                 # odd count
                        [8, 0.9, [0.1, float("nan"), 0.2, 0.1, 0.2, 0.2]],
                        [9, "x", [0.1, 0.1, 0.2, 0.1, 0.2, 0.2]],   # bad conf
                        [99, 0.9, [0.1, 0.1, 0.2, 0.1, 0.2, 0.2]],  # no box
                        "junk"])
    ts = tracks_from(boxes, masks)
    assert [t.track_id for t in ts] == [7, 8, 9]
    assert all(t.outline is None for t in ts)
    assert tracks_from(boxes, "{not json") == tracks_from(boxes, None)


def test_infinite_mask_id_and_undecodable_bytes_do_not_raise():
    boxes = json.dumps([[10, 20, 30, 40, 7]])
    (t,) = tracks_from(boxes, '[[Infinity, 0.9, null]]')
    assert (t.track_id, t.conf, t.outline) == (7, None, None)
    assert tracks_from(boxes, b"\xff\xfe[") == tracks_from(boxes, None)
