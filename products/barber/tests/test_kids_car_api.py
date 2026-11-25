from test_server import site_client


def test_no_place_at_first(tmp_path):
    c = site_client(tmp_path)
    r = c.get("/api/barber/kids-car").json()
    assert r["place"] is None and r["locator"] is False and "ph-1" in r["sizes"]


def test_a_manual_pin_and_clear(tmp_path):
    c = site_client(tmp_path)
    r = c.put("/api/barber/kids-car", json={"camera": "ph-1", "x": 0.5, "y": 0.5})
    assert r.status_code == 200
    p = r.json()["place"]
    assert p["camera"] == "ph-1" and p["source"] == "manual"
    assert abs((p["box"][0] + p["box"][2]) / 2 - 0.5) < 1e-6
    assert c.get("/api/barber/kids-car").json()["place"]["source"] == "manual"
    assert c.delete("/api/barber/kids-car").json()["place"] is None


def test_bad_pins_are_refused(tmp_path):
    c = site_client(tmp_path)
    assert c.put("/api/barber/kids-car", json={"camera": "nope", "x": 0.5, "y": 0.5}).status_code == 404
    assert c.put("/api/barber/kids-car", json={"camera": "ph-1", "x": 1.5, "y": 0.5}).status_code == 400
    assert c.put("/api/barber/kids-car", json={"camera": "ph-1"}).status_code in (400, 422)
