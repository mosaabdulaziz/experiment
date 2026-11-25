"""Which recorder a source gets. Pure, GPU-free, unit-tested.

Smart Video Record muxes the camera's own encoded stream -- NvDsSRInitParams'
width/height carry the comment "0 means no transcode" -- so it costs no NVENC
session. That matters because the deployment GPU (GeForce RTX 5090) caps
concurrent encoder sessions at a measured 8, and the encode branch spends one per
recorded stream: stream 9 simply fails to allocate an encoder.

SVR is RTSP-only. DS 9.0's nvurisrcbin says "Sources must be of type
source-type-rtsp" against both smart-record and smart-rec-container, so a file://
source keeps the encode branch and there is nothing to fall back to.

Exactly one recorder per source. Running both would double the cost this exists
to remove.
"""

VALID_OVERRIDES = ("auto", "svr", "encode", "none")


def recorder_kind(uri: str, override: str | None = None) -> str:
    """Return "svr", "encode", or "none" for one source.

    `override` comes from a source's optional `recorder:` key in sidecar.yaml;
    "auto" (or None) picks from the URI scheme. Forcing "svr" onto a non-RTSP
    source raises rather than producing a stream that records nothing.
    """
    if override is not None and override != "auto":
        if override not in VALID_OVERRIDES:
            raise ValueError(
                f"unknown recorder {override!r}; expected one of {VALID_OVERRIDES}")
        if override == "svr" and not _is_rtsp(uri):
            raise ValueError(
                f"recorder: svr requires an rtsp:// source, got {uri!r} -- Smart Video "
                "Record is RTSP-only in DeepStream, so this source would record nothing")
        return override
    return "svr" if _is_rtsp(uri) else "encode"


def _is_rtsp(uri: str) -> bool:
    return str(uri).lower().startswith("rtsp://")
