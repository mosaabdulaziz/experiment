from app.analyzer import _ThinkStripper


def _run(chunks):
    s = _ThinkStripper()
    out = "".join(s.feed(c) for c in chunks)
    return out + s.flush()


def test_passes_plain_text():
    assert _run(["hello ", "world"]) == "hello world"


def test_strips_whole_think_block():
    assert _run(["<think>reasoning</think>answer"]) == "answer"


def test_strips_think_split_across_chunks():
    assert _run(["before <thi", "nk>secret rea", "soning</thi", "nk> after"]) == "before  after"


def test_unterminated_think_is_suppressed_through_flush():
    # opens think but never closes — nothing after the tag should leak
    assert _run(["visible <think>dangling"]) == "visible "
