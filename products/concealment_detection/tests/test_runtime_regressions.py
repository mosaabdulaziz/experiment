from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from concealment.state_machine import ConcealmentMonitor
from concealment.types import DetectorSettings, MonitorStepResult
from config import load_config
from consumer import CameraWorker, Consumer, FrameQueue
from storage import ReviewStore, connect
from test_config_acl_storage_api import FakeRedis
from test_evidence_and_clock import _sample_candidate


@pytest.fixture
def store(tmp_path):
    result = ReviewStore(connect(tmp_path / 'review.db'))
    yield result
    result.close()


def entry(pts=1.0, epoch=1000.0):
    return {'id': f'{int(epoch * 1000)}-0', 'fields': {
        'frame_num': '1', 'pts': str(pts),
        'pose': json.dumps({'shape': [1920, 1080], 'items': []}),
    }}


class CandidateMonitor:
    settings = DetectorSettings()

    def __init__(self, tier='high'):
        self.tier = tier

    def update(self, frame):
        candidate = replace(_sample_candidate(self.tier, 'clothing'), pts=frame.pts)
        return MonitorStepResult(frame.camera, frame.pts, True, (candidate,), ())

    def invalidate(self, pts):
        pass


def test_queue_rejects_unbounded_capacity():
    with pytest.raises(ValueError):
        FrameQueue(0)


def test_malformed_entry_is_persisted_once_and_never_alerted(store):
    redis = FakeRedis()
    worker = CameraWorker('cam-1', ConcealmentMonitor('cam-1', DetectorSettings()), redis, store)
    bad = {'fields': {'pose': 'not-json', 'pts': '1'}}
    result = worker.process_entry(bad)
    worker.process_entry(bad)
    assert result['camera_ok'] is False
    rows = store.list_assessments()
    assert len(rows) == 1
    assert rows[0]['reason'] == 'malformed_pose'
    assert not redis.streams


def test_silent_camera_logs_once_and_recovers(store, monkeypatch):
    monkeypatch.setattr('consumer.time.monotonic', lambda: 10.0)
    worker = CameraWorker('cam-1', ConcealmentMonitor('cam-1', DetectorSettings()), None, store)
    worker.check_health(now=14.0)
    worker.check_health(now=15.0)
    assert len(store.list_assessments()) == 1
    assert worker.health['reason'] == 'stream_stale'
    worker.process_entry(entry())
    assert worker.health['camera_ok'] is True


def test_candidate_timestamp_comes_from_frame_not_processing_clock(store):
    worker = CameraWorker('cam-1', CandidateMonitor(), FakeRedis(), store)
    result = worker.process_entry(entry(epoch=1000.0))
    assert datetime.fromisoformat(result['candidates'][0]['display_time']).timestamp() == 1000.0


def test_candidates_do_not_request_svr_clips(store):
    for tier in ('low', 'high'):
        redis = FakeRedis()
        worker = CameraWorker('cam-1', CandidateMonitor(tier), redis, store)
        worker.process_entry(entry())
        assert 'iris:alert:cam-1' in redis.streams
        assert 'iris:svr_request:cam-1' not in redis.streams


def test_replay_uses_frame_lap_and_resets_on_wrap(store):
    worker = CameraWorker('cam-1', CandidateMonitor(), FakeRedis(), store,
                          footage_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
                          clip_duration_s=60.0)
    worker.set_replay_start(1000.0)
    first = worker.process_entry(entry(pts=59, epoch=1059))
    second = worker.process_entry(entry(pts=0, epoch=1060))
    assert first['candidates'][0]['display_time'].startswith('2020-01-01T00:00:59')
    assert second['candidates'][0]['display_time'].startswith('2020-01-01T00:00:00')


def test_runtime_lifespan_starts_and_stops_components(tmp_path):
    path = Path(__file__).parents[1] / 'main.py'
    assert path.exists(), 'Product has no executable backend entry point'
    spec = importlib.util.spec_from_file_location('concealment_main_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = load_config(env={'CONCEALMENT_DATA_DIR': str(tmp_path)})

    class Redis(FakeRedis):
        closed = False
        def ping(self):
            return True
        def close(self):
            self.closed = True

    redis = Redis()
    app = module.build_app(cfg, redis_client=redis)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        assert client.get('/status').json()['running'] is True
        assert 'cam-1' in client.get('/status').json()['cameras']
        assert client.get('/status').json()['capabilities']['cart_detection'] is False
    assert redis.closed
    assert not app.state.runtime.consumer.alive
    assert not any(w.alive for w in app.state.runtime.workers.values())


def test_launch_manifest_declares_backend_dependencies_and_sidecar():
    import tomllib
    import yaml
    product = Path(__file__).parents[1]
    manifest_path = product / 'pyproject.toml'
    assert manifest_path.is_file(), 'Backend dependencies have no installation manifest'
    manifest = tomllib.loads(manifest_path.read_text(encoding='utf-8'))
    dependencies = manifest['project']['dependencies']
    for package in ('redis', 'fastapi', 'uvicorn', 'pyyaml', 'numpy'):
        assert any(dep.startswith(package) for dep in dependencies)
    compose = yaml.safe_load((product.parents[1] / 'docker-compose.yml').read_text(encoding='utf-8'))
    sidecar = compose['services'].get('deepstream-sidecar-concealment')
    assert sidecar is not None
    assert 'concealment' in sidecar['profiles']
    assert 'IRIS_SIDECAR_ROLE=concealment_detection' in sidecar['environment']
