"""Moving the character and aiming the camera.

This is the second dispatch channel, alongside the timeline. It exists because a `Plan` is
a discrete schedule of presses and these two things are not discrete: movement is a
continuously corrected held state, and camera aim is a slewing analog target.

The camera gets equal billing with movement rather than being treated as a view setting,
for two reasons that both bite hard:

* **The camera is an actuator.** In a 3D game you only perceive what is in the frustum. A
  tower behind you is not absent, it is unobserved, and the fix is to turn and look.
  Perception here is steerable, and steering it is an action with a cost.
* **The camera is the movement reference frame.** `W` means "away from the camera", so
  turning to look at something changes what every movement key does. The two cannot be
  controlled independently, which is why one object owns both.
"""

from .camera import CameraController, CameraIntent, CameraMode
from .channel import NavigationChannel, NavigationState
from .gaze import GazePolicy, GazeRequest, GazeTarget
from .movement import MovementController, MovementIntent

__all__ = [
    "CameraController",
    "CameraIntent",
    "CameraMode",
    "GazePolicy",
    "GazeRequest",
    "GazeTarget",
    "MovementController",
    "MovementIntent",
    "NavigationChannel",
    "NavigationState",
]
