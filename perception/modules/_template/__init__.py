"""Copy this folder to perception/modules/<your_capability>/, rename the class and `name`, and delete the hooks you
do not need. Then switch it on in a product's sidecar yaml:

    modules:
      <your_capability>:
        threshold: 3

Rules: no pyds in here (use FrameView / ObjectView; metadata a module needs comes through `extract` and
perception.engine.nvds); never block in a hook; never call another module (read what an
earlier module left on the ObjectViews, and say so in `requires`); put every tunable in `Settings`.
"""
from dataclasses import dataclass

from perception.engine.frame import FrameOutput, FrameView
from perception.engine.modules import Module


@dataclass
class Settings:
    threshold: int = 3                   # a setting with a default is optional in the yaml; without one it is required


class Template(Module):
    name = "_template"                   # must equal the yaml key and the folder name
    Settings = Settings
    requires = ()                        # modules that must be listed before this one

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        busy = [o for o in frame.objects if o.label == "person"]
        if len(busy) >= self.settings.threshold:
            out.fields[self.name] = {"count": len(busy)}                # lands in iris:meta under this module's name
            # An alert the app turns into an event (saved, broadcast, clip) with no code of its own for this module:
            self.services.alert(frame.stream_id, "busy_room", severity="info", category="operations", pts=frame.pts,
                                count=len(busy))


MODULE = Template
