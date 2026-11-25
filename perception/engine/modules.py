"""The module contract: how use-case logic plugs into the engine without the engine knowing it exists.

A module is a package perception/modules/<name>/ whose __init__ exposes `MODULE`, a subclass of `Module`. The
engine reads `modules:` from the sidecar yaml -- an ordered mapping name -> settings -- imports each named
package, builds its Settings from the mapping, and calls its hooks in that order. Nothing in the engine imports
a module by name; a product turns a capability on by naming it in its yaml and tunes it with its settings.

Hook order for one frame (all on the streaming thread unless noted; keep them cheap, never block on I/O):

    sample          decide whether this frame is a meta sample (first module with an opinion wins)
    extract         a function that reads what this module needs off each object's metadata into `obj.attrs`
                    (asked when the frame's objects are first read, so it can depend on the frame; return None to skip)
    on_frame        every frame, before any gating     -- alerting that must see every frame
    transform       frames that will be published      -- change `frame.objects` (filter, replace, add, relabel)
    wants_pixels    ask for the CPU copy of the frame
    on_pixels       given the CPU copy                 -- crops, samplers
    analyze         with objects final (and pixels)    -- events, annotations, extra meta via FrameOutput; it can
                    register `out.after_publish(fn)` to get the clean frame on the publisher thread, after drawing

A module never calls another module. Data passes through `ObjectView` fields and `FrameOutput` only.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .frame import FrameOutput, FrameView, ObjectView
from .model_manifest import ModelRef

OVERLAY_MAXLEN = 1500   # iris:overlay:<camera> entries kept: ~3.5 min at 7 analysed frames/s; clips come 20-30 s late

logger = logging.getLogger("iris.sidecar")

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _camera_key_id(stream_id: str) -> str:
    return stream_id


class ModuleConfigError(ValueError):
    """The `modules:` section is wrong; the engine refuses to start rather than run half-configured."""


class ModuleStartupError(RuntimeError):
    """A module could not set itself up; the engine refuses to start rather than run without it."""


@dataclass
class GieSpec:
    """An inference element a module needs the engine to put in the chain."""
    name: str                            # unique element name
    config_path: str
    role: str = "secondary"              # "primary" runs on the full frame, "secondary" on tracked objects
    interval: int = 0
    engine_path: Optional[str] = None    # checked for uniqueness across all GIEs
    # True: the element's batch size is the pipeline's (a detector run on every frame). False: the batch size is the
    # one in its own config file -- a secondary detector sees a variable number of objects per frame and its engine
    # is built for that size.
    batch_from_mux: bool = True


@dataclass
class PipelineHandle:
    """What `on_built` is given: the built pipeline, and `get(name)` to look up one of its elements by name."""
    pipeline: Any
    get: Callable[[str], Any]


@dataclass
class ElementSpec:
    """A GStreamer element (not an inference one) a module needs in the chain, after every inference element."""
    name: str
    factory: str                         # the GStreamer element factory, e.g. "nvdsanalytics"
    properties: dict = field(default_factory=dict)


@dataclass
class EngineServices:
    """The only thing a module gets from the engine. Everything is a plain callable or value, so a module can be
    tested with a few stand-ins and no GStreamer."""
    redis: Any
    redis_factory: Callable[[], Any]
    emit: Callable[..., None]                           # emit(stream_key, fields, maxlen=64)
    request_clip: Callable[..., Any]                    # request_clip(stream_id, now, age=None)
    stream_ids: list
    timer: Callable[[float, Callable[[], Any]], None]   # timer(interval_seconds, fn)
    telemetry: Any
    paths: dict = field(default_factory=dict)
    batch_size: int = 1                                 # the inference batch size every GIE is built with
    boot_id: str = ""
    config: dict = field(default_factory=dict)          # the whole engine config, read-only by convention
    states: dict = field(default_factory=dict)          # stream id -> the engine's per-stream state
    mux_size: tuple = (0, 0)                            # (width, height) detections are reported in
    engine: Any = None                                  # the engine's own settings (rates, sizes, preview), read-only
    index_to_id: dict = field(default_factory=dict)     # streammux pad index -> stream id
    sources: list = field(default_factory=list)         # the sources THIS process owns, in streammux pad order
    all_sources: list = field(default_factory=list)     # every configured source, before role/shard filtering
    role: Optional[str] = None                          # the source-tag this process serves (None = every source)
    shard_id: int = 0                                   # IRIS_SHARD_ID: replicas of one config differ only by this
    # stream id -> the id its Redis keys carry: the camera id, or `<camera>@<role>` with role_scoped_keys. A module
    # that writes a per-camera stream itself names it with this: f"iris:zone_alert:{services.key_id(stream_id)}".
    key_id: Callable[[str], str] = _camera_key_id

    def alert(self, stream_id: str, kind: str, *, message: str = "", severity: str = "warning",
              category: str = "security", pts: Optional[float] = None, zone: str = "", count: Optional[int] = None,
              **extra: Any) -> bool:
        """Raise an alert on a camera: the one way a module reaches the app's alert flow. Written to
        `iris:alert:<stream_id>`, where the app turns it into an event (saved, broadcast, clip queued) with no code of
        its own for the module. `kind` names it ("proximity_warning"; it is the event's tag and, with no `message`,
        its text), `severity` is critical|warning|info, `category` safety|security|storage|operations, `pts` the
        frame's time in seconds (the clip is centred on it), `zone` an id from the app's zones, `count` a number the
        event carries. `extra` fields are kept on the stream entry for consumers that want them. True when written."""
        fields = {"v": "1", "kind": kind, "severity": severity, "category": category, "message": message,
                  "zone": zone, "pts": "" if pts is None else f"{pts:.3f}", "count": "" if count is None else str(count)}
        fields.update({k: str(v) for k, v in extra.items() if v is not None})
        return self.emit(f"iris:alert:{self.key_id(stream_id)}", fields, maxlen=256)

    def overlay(self, stream_id: str, module: str, pts: float, size: tuple, shapes: list, at: float = 0.0) -> bool:
        """What a module wants drawn on this camera's recorded clips at frame time `pts` (s): `shapes` in a
        `size` = (width, height) coordinate space, each {"type": "line", "p1", "p2", "label", "style"} or
        {"type": "box", "rect": [left, top, width, height], "style", "id"}. Written to `iris:overlay:<stream_id>`,
        which the app's clip builder draws (app/overlay_track.py); about 3 minutes are kept. True when written.
        `at`: the frame's arrival (FrameView.arrived, wall s), which places the entry on a smart-record clip."""
        fields = {"v": "1", "module": module, "pts": f"{pts:.3f}", "w": str(int(size[0])), "h": str(int(size[1])),
                  "shapes": json.dumps(shapes, separators=(",", ":"))}
        if at:
            fields["at"] = f"{at:.3f}"
        return self.emit(f"iris:overlay:{self.key_id(stream_id)}", fields, maxlen=OVERLAY_MAXLEN)

    def state(self, module: str, stream_id: str) -> dict:
        """Per-module, per-stream scratch dict (replaces attributes bolted onto the stream state)."""
        st = self.states.get(stream_id)
        if st is None:
            return {}
        return st.module_state.setdefault(module, {})


class Module:
    """Base class: every hook is a no-op, so a module overrides only what it needs."""
    name: str = ""
    requires: tuple = ()                 # names of modules that must also be configured (checked at startup)
    after: tuple = ()                    # modules that, WHEN configured, must be listed before this one
    Settings: Optional[type] = None      # a dataclass; unknown keys in the yaml refuse startup
    # True when the module's own workers write per-camera Redis streams named by the camera id rather than
    # services.key_id(stream_id): such a module cannot run with role_scoped_keys (the engine refuses it there).
    camera_keyed_streams: bool = False

    def __init__(self, settings: Any, services: EngineServices) -> None:
        self.settings = settings
        self.services = services

    @classmethod
    def model_files(cls, settings: Any, every_variant: bool = False) -> list:
        """The model files these settings make the module load, as ModelRefs (perception/engine/model_manifest.py):
        plain files, nvinfer configs (the engine reads their onnx-file/labelfile-path) and model-pack directories.
        `via` names the setting, e.g. "staff_classifier"; the engine prefixes `modules.<name>.`. By default only what
        THESE settings load (a switch picks one model or another); with `every_variant`, what any value of such a
        switch would load, so a manifest can cover the fallback too. Called on Settings alone, before any module is
        constructed (startup model manifest check, its CI completeness test, the manifest generator); it must not
        load anything."""
        return []

    @property
    def overlay_boxes(self):
        """This module's writer of the boxes its alerts' clips draw (perception/lib/overlay_boxes.py), made on first
        use: `self.overlay_boxes.write(stream_id, pts, [box(...), ...])`."""
        writer = self.__dict__.get("_overlay_boxes")
        if writer is None:
            from perception.lib.overlay_boxes import OverlayWriter
            writer = self.__dict__["_overlay_boxes"] = OverlayWriter(self.services, self.name)
        return writer

    # build time ------------------------------------------------------------------------------------------
    def gies(self) -> list:
        return []

    def elements(self) -> list:
        """Non-inference elements to add after the inference chain, as ElementSpecs."""
        return []

    def on_built(self, pipeline: Any) -> None:
        """The pipeline exists: look up this module's elements by name, add pad probes."""

    def start(self) -> None:
        """The pipeline is about to play: start worker threads."""

    def stop(self) -> None:
        """Shutdown: stop worker threads and flush."""

    # hot path --------------------------------------------------------------------------------------------
    def sample(self, frame: FrameView) -> Optional[bool]:
        """True/False overrides the engine's time-based sampling clock for this frame; None has no opinion."""
        return None

    def extract(self, frame: FrameView) -> Optional[Callable[[ObjectView], None]]:
        """A function the engine calls on every object of this frame, while its pyds metadata is valid, to fill
        `obj.attrs` (read through perception.engine.nvds). None: this module needs nothing read this frame. It runs
        once per object, so keep it to a few reads."""
        return None

    def on_frame(self, frame: FrameView, out: FrameOutput) -> None:
        pass

    def transform(self, frame: FrameView) -> None:
        pass

    def wants_pixels(self, frame: FrameView) -> bool:
        return False

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        pass

    def on_pixels(self, frame: FrameView, bgr: Any) -> None:
        pass

    # optional --------------------------------------------------------------------------------------------
    def control(self) -> dict:
        """{redis control key: handler(value)} polled by the engine's control loop."""
        return {}

    def gauges(self) -> dict:
        """{gauge name: callable returning a number} registered with telemetry."""
        return {}


def build_settings(cls: Optional[type], raw: Any, module: str) -> Any:
    """Make a module's Settings from its yaml mapping. Unknown keys and missing required keys are errors."""
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ModuleConfigError(f"modules.{module}: expected a mapping of settings, got {type(raw).__name__}")
    if cls is None:
        if raw:
            raise ModuleConfigError(f"modules.{module}: takes no settings, got {sorted(raw)}")
        return None
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ModuleConfigError(f"modules.{module}: unknown setting(s) {unknown}; known: {sorted(fields)}")
    missing = sorted(n for n, f in fields.items()
                     if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and n not in raw)
    if missing:
        raise ModuleConfigError(f"modules.{module}: missing required setting(s) {missing}")
    return cls(**raw)


def camera_keyed_modules(modules) -> list:
    """The loaded modules that write camera-keyed streams (Module.camera_keyed_streams), by name."""
    return [m.name for m in modules.modules if getattr(m, "camera_keyed_streams", False)]


class ModuleSet:
    """The configured modules, in order, with a failure in one never stopping the pipeline or the others."""

    LOG_EVERY = 10.0                     # seconds between repeats of the same module/hook failure

    def __init__(self, modules: list) -> None:
        self.modules = list(modules)
        self._last_logged: dict = {}
        self.failures: dict = {}         # (module, hook) -> count

    def __bool__(self) -> bool:
        return bool(self.modules)

    def names(self) -> list:
        return [m.name for m in self.modules]

    def get(self, name: str) -> Optional[Module]:
        return next((m for m in self.modules if m.name == name), None)

    def _fail(self, module: Module, hook: str, exc: Exception) -> None:
        key = (module.name, hook)
        self.failures[key] = self.failures.get(key, 0) + 1
        now = time.monotonic()
        if now - self._last_logged.get(key, -1e9) >= self.LOG_EVERY:
            self._last_logged[key] = now
            logger.error("module %s.%s failed (%d so far): %s", module.name, hook, self.failures[key], exc,
                         exc_info=True)

    def run(self, hook: str, *args: Any) -> None:
        for m in self.modules:
            try:
                getattr(m, hook)(*args)
            except Exception as exc:  # noqa: BLE001 -- a module bug must not take the probe down
                self._fail(m, hook, exc)

    def setup(self, hook: str, *args: Any) -> None:
        """Run a build-time hook (`on_built`, `start`) on every module. Unlike the per-frame hooks, a failure here
        is fatal: a module that could not set itself up would silently do nothing for the whole run."""
        for m in self.modules:
            try:
                getattr(m, hook)(*args)
            except Exception as exc:  # noqa: BLE001
                raise ModuleStartupError(f"module {m.name}.{hook} failed: {exc}") from exc

    def any(self, hook: str, *args: Any) -> bool:
        """True if any module's predicate hook says yes. Every module is asked (they may keep state)."""
        result = False
        for m in self.modules:
            try:
                result = bool(getattr(m, hook)(*args)) or result
            except Exception as exc:  # noqa: BLE001
                self._fail(m, hook, exc)
        return result

    def collect(self, hook: str, *args: Any) -> list:
        """Every module's non-None answer to a hook, in module order."""
        answers = []
        for m in self.modules:
            try:
                answer = getattr(m, hook)(*args)
            except Exception as exc:  # noqa: BLE001
                self._fail(m, hook, exc)
                continue
            if answer is not None:
                answers.append(answer)
        return answers

    def first(self, hook: str, *args: Any) -> Any:
        """The first non-None answer from a hook, in module order (None if nobody has an opinion)."""
        for m in self.modules:
            try:
                answer = getattr(m, hook)(*args)
            except Exception as exc:  # noqa: BLE001
                self._fail(m, hook, exc)
                continue
            if answer is not None:
                return answer
        return None

    def gies(self) -> list:
        return [g for m in self.modules for g in m.gies()]

    def elements(self) -> list:
        return [e for m in self.modules for e in m.elements()]

    def controls(self) -> dict:
        """{control key: handler}. Several modules may listen to one key (the zones version: the zones module and every
        module that reads the zones file): its handler calls each of theirs in module order, and re-raises the first failure
        once all were called, so the poller retries the change (perception/engine/control.py) -- a reload twice is
        harmless, one listener's failure silencing another is not."""
        listeners: dict = {}
        for m in self.modules:
            for key, handler in m.control().items():
                listeners.setdefault(key, []).append(handler)
        return {key: hs[0] if len(hs) == 1 else _fan_out(hs) for key, hs in listeners.items()}


def _fan_out(handlers: list):
    def handle(value):
        failed = None
        for handler in handlers:
            try:
                handler(value)
            except Exception as exc:  # noqa: BLE001 -- every listener hears it; the first failure is re-raised
                failed = failed or exc
        if failed is not None:
            raise failed
    return handle


def _module_classes(config: dict, package: str):
    """(name, raw settings, Module subclass) for every module named in `modules:`, in order."""
    for name, raw in config.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ModuleConfigError(f"modules: {name!r} is not a valid module name (lowercase letters, digits, _)")
        try:
            pkg = importlib.import_module(f"{package}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{package}.{name}":
                raise ModuleConfigError(f"modules.{name}: no such module ({package}.{name})") from exc
            raise
        cls = getattr(pkg, "MODULE", None)
        if not (isinstance(cls, type) and issubclass(cls, Module)):
            raise ModuleConfigError(f"modules.{name}: package {package}.{name} must expose MODULE, a Module subclass")
        if cls.name != name:
            raise ModuleConfigError(f"modules.{name}: package exposes a module named {cls.name!r}")
        yield name, raw, cls


def declared_model_files(config: Optional[dict], package: str = "perception.modules", every_variant: bool = False) -> list:
    """Every ModelRef the modules named in `modules:` would load with these settings, without constructing them."""
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ModuleConfigError(f"`modules:` must be a mapping of name -> settings, got {type(config).__name__}")
    refs = []
    for name, raw, cls in _module_classes(config, package):
        for ref in cls.model_files(build_settings(cls.Settings, raw, name), every_variant):
            refs.append(ModelRef(ref.path, f"modules.{name}.{ref.via}", ref.kind))
    return refs


def load_modules(config: Optional[dict], services: EngineServices, package: str = "perception.modules") -> ModuleSet:
    """Import and construct every module named in `modules:`, in the order given."""
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ModuleConfigError(f"`modules:` must be a mapping of name -> settings, got {type(config).__name__}")
    loaded = [cls(build_settings(cls.Settings, raw, name), services)
              for name, raw, cls in _module_classes(config, package)]
    present = {m.name for m in loaded}
    for m in loaded:
        absent = [r for r in m.requires if r not in present]
        if absent:
            raise ModuleConfigError(f"modules.{m.name}: requires {absent}, which {'is' if len(absent) == 1 else 'are'} "
                                    f"not configured")
        # a requirement must also come EARLIER in the order, since hooks run in that order and data flows forward
        order = [x.name for x in loaded]
        late = [r for r in m.requires if order.index(r) > order.index(m.name)]
        if late:
            raise ModuleConfigError(f"modules.{m.name}: requires {late}, which must be listed before it")
        # `after` is the same rule for a module that merely reads what another leaves behind when it is there
        misordered = [r for r in m.after if r in present and order.index(r) > order.index(m.name)]
        if misordered:
            raise ModuleConfigError(f"modules.{m.name}: must be listed after {misordered}")
    return ModuleSet(loaded)
