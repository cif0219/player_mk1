"""FantCraft — the first game we test that we also *own*.

FantCraft is the in-house voxel MMORPG (D:/projects/fancraft). Owning the game
changes the rules of engagement completely, and this profile leans into all of it:

* **No ToS problem.** It is our game; automating it is the point. This profile
  exists to *playtest* FantCraft — the player is the tester, and the game ships
  a ground-truth oracle (`docs/PLAYTEST.md` over there) purpose-built to score
  this player's perception and decisions against what the server actually knew.

* **No hand calibration.** The FantCraft client publishes a HUD manifest —
  measured off its live DOM — with every bar and hotbar slot as viewport
  fractions, the bar palette, and which ability sits on which key. The layout
  here is *loaded* from that manifest, not drawn by hand, so it cannot drift.

* **Ground truth exists.** The server records what it believed (truth.jsonl),
  the client records what it rendered (client.jsonl), and this player records
  what it perceived (sessions/). `tools/fancraft_report.py` joins the three and
  says which layer got something wrong. That closes the loop no commercial game
  ever lets us close.
"""

from .profile import FancraftConfig, build, build_api, build_screen, config_from_dict

__all__ = ["FancraftConfig", "build", "build_api", "build_screen", "config_from_dict"]
