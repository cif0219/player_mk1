"""API control: play through a game's agent gateway instead of its screen.

The screen path is capture → probes → `WorldState` → policy → keystrokes. This
package keeps the middle of that pipeline — `WorldState`, reflexes, the
deadline timeline, the guard chain — and swaps the two ends:

* `ApiSource` is a `CaptureSource` whose frames carry a structured snapshot
  from the game's gateway instead of pixels; a `SnapshotSensor` turns that
  snapshot into fields.
* `ApiBackend` is an `InputBackend` whose "keys" are intents (walk, face,
  guard, skill) the gateway relays to the game as ordinary client messages.

Nothing about the game is privileged: the gateway is a normal game client and
the server referees every rule. What the tester gains is exact positions and
telegraph timings in place of a probe's guess — which is what makes a
150 ms parry window playable — and what it loses is the interface test. Both
matter, so the screen path stays; this is the second way in, not a replacement.
"""

from .backend import ApiBackend, KeyActions
from .gateway import GatewayClient, GatewayError, Snapshot
from .sensor import SnapshotSensor
from .source import ApiFrame, ApiSource
from .transport import ApiTransport

__all__ = [
    "ApiBackend",
    "ApiFrame",
    "ApiSource",
    "ApiTransport",
    "GatewayClient",
    "GatewayError",
    "KeyActions",
    "Snapshot",
    "SnapshotSensor",
]
