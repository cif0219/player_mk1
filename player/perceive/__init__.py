"""Sensors: pixels in, `WorldState` out.

Two families, for two different jobs:

* **Probes** read fixed calibrated regions with numpy. HP bars, cast bars, cooldown
  rings. Fast, exact, no training data. A model would be strictly worse at these.
* **Detectors** find things at unknown positions — ground telegraphs, nameplates. This is
  where ML earns its place, and nowhere else.
"""

from .detector import Detection, DetectorSensor, TelegraphSegmenter
from .pipeline import PerceptionPipeline
from .probes import (
    BarProbe,
    ColorProbe,
    CooldownRingProbe,
    PresenceProbe,
    Probe,
    ProbeReading,
    TemplateProbe,
)
from .sensor import ProbeSensor, Sensor, SensorBundle

__all__ = [
    "BarProbe",
    "ColorProbe",
    "CooldownRingProbe",
    "Detection",
    "DetectorSensor",
    "PerceptionPipeline",
    "PresenceProbe",
    "Probe",
    "ProbeReading",
    "ProbeSensor",
    "Sensor",
    "SensorBundle",
    "TelegraphSegmenter",
    "TemplateProbe",
]
