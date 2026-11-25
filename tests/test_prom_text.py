from app.prom_text import render


def test_render_types_labels_and_escaping():
    out = render([("iris_x_total", {}, 3), ("iris_y", {"cam": 'a"b'}, 1.5), ("iris_y", {"cam": "c"}, 0)])
    assert "# TYPE iris_x_total counter\niris_x_total 3\n" in out
    assert out.count("# TYPE iris_y gauge") == 1
    assert 'iris_y{cam="a\\"b"} 1.5' in out and 'iris_y{cam="c"} 0' in out
