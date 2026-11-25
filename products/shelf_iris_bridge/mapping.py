"""The mapping file: which iris-platform pipeline x camera each DeepStream camera's shelf region reports to.

    # <DeepStream camera (source id, the X of iris:alert:X, or of iris:alert:X@<role> with role_scoped_keys)>:
    #   <region (the shelf_void roi name: R1, R2, ... or its own name)>: {pipelineId, cameraId}
    cam-shelf-test:
      R1: {pipelineId: 1f9c0b8e-..., cameraId: b3d1c0de-...}
      R2: {pipelineId: 5a7e..., cameraId: b3d1c0de-...}

Both ids are iris-platform's (the pipeline's id and the camera's id, UUIDs). Several regions may report to the same
pipeline x camera: each region is its own situation (its own correlationKey).
"""
import re

import yaml

UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class MappingError(ValueError):
    """The mapping file can't be used: the bridge refuses to start."""


class Mapping:
    def __init__(self, raw):
        if not isinstance(raw, dict) or not raw:
            raise MappingError("the mapping must name at least one DeepStream camera")
        self.targets = {}                 # (camera, region) -> (pipelineId, cameraId)
        for cam, regions in raw.items():
            if not isinstance(regions, dict) or not regions:
                raise MappingError(f"{cam}: map at least one region to {{pipelineId, cameraId}}")
            for region, t in regions.items():
                if not isinstance(t, dict) or sorted(t) != ["cameraId", "pipelineId"]:
                    raise MappingError(f"{cam}/{region}: needs exactly pipelineId and cameraId")
                for key in ("pipelineId", "cameraId"):
                    if not isinstance(t[key], str) or not UUID.match(t[key]):
                        raise MappingError(f"{cam}/{region}: {key} must be an iris-platform id (UUID)")
                self.targets[(str(cam), str(region))] = (t["pipelineId"].lower(), t["cameraId"].lower())

    @classmethod
    def load(cls, path: str) -> "Mapping":
        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except OSError as e:
            raise MappingError(f"cannot read {path}: {e.strerror}") from e
        except yaml.YAMLError as e:
            raise MappingError(f"{path} is not valid YAML: {e}") from e
        return cls(raw)

    @property
    def cameras(self) -> list:
        """The DeepStream cameras whose alert streams the bridge reads."""
        return sorted({cam for cam, _ in self.targets})

    def target(self, camera: str, region: str):
        """(pipelineId, cameraId) for a DeepStream camera's region, or None."""
        return self.targets.get((camera, region))

    def regions_of(self, pipeline_id: str, camera_id: str) -> list:
        """The [(DeepStream camera, region)] reporting to one iris-platform pipeline x camera."""
        key = (pipeline_id.lower(), camera_id.lower())
        return sorted(src for src, t in self.targets.items() if t == key)
