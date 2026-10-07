"""Turning waymark sightings into "where am I and which way am I facing".

The key result this module exists to exploit: **two identified waymarks on the minimap
give position, scale, and camera yaw simultaneously.** Fitting a similarity between the
known arena layout and where those marks appear recovers all three at once, so nothing has
to be hand-calibrated — not the minimap's pixels-per-metre, not whether the minimap
rotates with the camera or is locked to north, not the camera's field of view.

That matters beyond convenience. A hand-measured constant is silently wrong the moment
someone changes a UI scale or a zoom level, and being silently wrong about which way you
are facing is how a player walks into the thing it was trying to dodge.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from ..clock import now
from .arena import (
    ArenaFrame,
    ArenaPoint,
    CameraPose,
    Observability,
    WaymarkLayout,
    normalise_deg,
)
from .transform import Homography, Similarity2D, fit_homography, fit_similarity


@dataclass(frozen=True, slots=True)
class Sighting:
    """A landmark we spotted, and where it appeared.

    `minimap_offset` is in minimap pixels relative to the minimap centre, in screen
    convention (+y down). `view_point` is in frame pixels in the 3D view, when the mark
    was also visible there.

    `candidates` exists because identification is genuinely ambiguous. FFXIV's waymarks
    come in two families that **share colours** — A and 1 are both red, B and 2 both
    yellow, and so on — so a colour-keyed detector can narrow a blob to two possibilities
    and no further. Rather than guessing, a sighting carries both and lets geometry settle
    it: only one assignment fits a consistent rigid transform, and the fit residual says
    which by an enormous margin.
    """

    mark_id: str
    minimap_offset: tuple[float, float] | None = None
    view_point: tuple[float, float] | None = None
    confidence: float = 1.0
    candidates: tuple[str, ...] = ()

    def options(self) -> tuple[str, ...]:
        return self.candidates if self.candidates else (self.mark_id,)

    def identified_as(self, mark_id: str) -> "Sighting":
        return Sighting(
            mark_id=mark_id,
            minimap_offset=self.minimap_offset,
            view_point=self.view_point,
            confidence=self.confidence,
            candidates=(),
        )


@dataclass(slots=True)
class LocalizationResult:
    """Where we think we are, how sure we are, and how we worked it out."""

    player: ArenaPoint | None = None
    pose: CameraPose = field(default_factory=CameraPose)
    scale_px_per_m: float = 0.0
    homography: Homography | None = None
    confidence: float = 0.0
    method: str = "none"
    residual_px: float = 0.0
    sightings_used: int = 0
    # True when more than one mark assignment fitted equally well — a symmetric waymark
    # layout. Downstream this is a reason to be careful, not a reason to stop.
    ambiguous: bool = False
    reason: str = ""
    at: float = 0.0

    @property
    def usable(self) -> bool:
        return self.player is not None and self.confidence > 0.0

    def to_frame(self, layout: WaymarkLayout, usable_fov_deg: float = 70.0) -> ArenaFrame | None:
        if self.player is None:
            return None
        return ArenaFrame(
            player=self.player,
            pose=self.pose,
            layout=layout,
            observability=Observability(
                pose=self.pose, player=self.player, usable_fov_deg=usable_fov_deg
            ),
            scale_px_per_m=self.scale_px_per_m,
            confidence=self.confidence,
            source=self.method,
        )

    def describe(self) -> str:
        if not self.usable:
            return f"localize: none ({self.reason or 'no sightings'})"
        return (
            f"localize: ({self.player.x:+.1f}, {self.player.y:+.1f})m "
            f"yaw={self.pose.yaw_deg:.0f}deg "
            f"scale={self.scale_px_per_m:.2f}px/m "
            f"resid={self.residual_px:.2f}px "
            f"n={self.sightings_used} conf={self.confidence:.2f} [{self.method}]"
        )


class Localizer:
    """Fits arena position and camera yaw from recognised landmarks.

    Two independent paths, deliberately kept separate rather than blended:

    * **Minimap similarity** — cheap, always available while the minimap is on screen,
      accuracy bounded by minimap resolution.
    * **View homography** — needs four waymarks visible in the 3D view, but gives full
      ground-plane projection so screen-space detections become arena positions.

    They are not averaged. The minimap answer is the fallback and the homography is
    preferred when available, because averaging two estimates with very different error
    characteristics produces something with the failure modes of both.
    """

    def __init__(
        self,
        layout: WaymarkLayout,
        *,
        max_residual_px: float = 6.0,
        min_sightings: int = 2,
        max_hypotheses: int = 256,
        ambiguity_tolerance_px: float = 0.5,
    ) -> None:
        self.layout = layout
        self.max_residual_px = max_residual_px
        self.min_sightings = max(2, min_sightings)
        # Ceiling on the ambiguity search. This runs on the decide thread every frame, so
        # a pathological frame must degrade rather than stall the loop.
        self.max_hypotheses = max_hypotheses
        # Two assignments fitting this closely are treated as indistinguishable by
        # geometry alone, and settled by temporal continuity instead.
        self.ambiguity_tolerance_px = ambiguity_tolerance_px
        self.last: LocalizationResult = LocalizationResult()
        self.last_was_ambiguous = False
        self.rejections = 0
        self.solves = 0
        self.ambiguous_frames = 0

    def set_layout(self, layout: WaymarkLayout) -> None:
        self.layout = layout
        self.last = LocalizationResult()

    # -- minimap path ------------------------------------------------------------

    def from_minimap(self, sightings: list[Sighting], at: float | None = None) -> LocalizationResult:
        at = now() if at is None else at
        candidates = [s for s in sightings if s.minimap_offset is not None]
        if len(candidates) < self.min_sightings:
            return self._fail(f"need {self.min_sightings} marks, saw {len(candidates)}", at)

        resolved = self._disambiguate(candidates)
        if resolved is None:
            return self._fail("no consistent assignment of ambiguous marks", at)
        usable, fit = resolved

        if len(usable) < self.min_sightings:
            return self._fail(f"need {self.min_sightings} marks, saw {len(usable)}", at)
        if fit is None or not fit.valid:
            return self._fail("degenerate minimap geometry (marks collinear or coincident)", at)

        if fit.residual_rms > self.max_residual_px:
            # A misidentified mark, or an icon clipped at the minimap edge. Either way,
            # reporting a confident wrong position is much worse than reporting none, so
            # this fails rather than degrades.
            self.rejections += 1
            return self._fail(
                f"residual {fit.residual_rms:.1f}px over {self.max_residual_px:.1f}px "
                f"— probable mark misidentification",
                at,
            )

        # The player sits at the minimap centre, so their arena position is whatever maps
        # onto the origin.
        player = fit.unapply(ArenaPoint(0.0, 0.0))
        yaw = normalise_deg(fit.rotation_deg)
        confidence = _confidence(fit.residual_rms, self.max_residual_px, len(usable))

        ambiguous = self.last_was_ambiguous
        reason = ""
        if ambiguous:
            self.ambiguous_frames += 1
            had_history = self.last.usable
            # A tie broken by history is trustworthy; a tie broken by nothing is a coin
            # flip, and the caller deserves to know which it got.
            confidence = confidence * (0.8 if had_history else 0.35)
            reason = (
                "symmetric layout; resolved by continuity"
                if had_history
                else "symmetric layout; no history to disambiguate"
            )

        self.solves += 1
        self.last = LocalizationResult(
            player=player,
            pose=CameraPose(yaw_deg=yaw, confidence=confidence),
            scale_px_per_m=fit.scale,
            confidence=round(confidence, 4),
            method="minimap",
            residual_px=fit.residual_rms,
            sightings_used=len(usable),
            ambiguous=ambiguous,
            reason=reason,
            at=at,
        )
        return self.last

    def _disambiguate(
        self, sightings: list[Sighting]
    ) -> tuple[list[Sighting], Similarity2D | None] | None:
        """Resolve ambiguous marks by picking the assignment that actually fits.

        FFXIV waymark families share colours, so a colour-keyed detector narrows each blob
        to two possibilities. Geometry settles it: only the correct assignment admits a
        consistent rotation-and-scale, and the residual separates right from wrong by
        orders of magnitude — an exact fit versus a dozen pixels. That gap is what makes
        this reliable rather than a coin flip.

        The search is **group-wise**, not over all sightings at once. Blobs of the same
        colour compete only for that colour's marks, so four ambiguous pairs is sixteen
        hypotheses rather than two hundred and fifty-six. That difference is what keeps it
        inside a 60Hz decide loop.
        """
        groups: dict[tuple[str, ...], list[Sighting]] = {}
        for sighting in sightings:
            groups.setdefault(sighting.options(), []).append(sighting)

        per_group: list[list[list[Sighting]]] = []
        total = 1
        for marks, members in groups.items():
            available = [m for m in marks if self.layout.get(m) is not None]
            if not available:
                continue
            choices = _injective_assignments(members, available)
            if not choices:
                continue
            per_group.append(choices)
            total *= len(choices)
            if total > self.max_hypotheses:
                # Too ambiguous to search honestly. Keep only the groups that resolve on
                # their own rather than guessing at the rest.
                per_group = [g for g in per_group if len(g) == 1]
                break

        if not per_group:
            return None

        hypotheses: list[tuple[list[Sighting], Similarity2D]] = []
        for combination in itertools.product(*per_group):
            chosen = [s for group in combination for s in group]
            # A mark cannot be in two places at once.
            if len({s.mark_id for s in chosen}) != len(chosen):
                continue
            fitted = self._fit_assignment(chosen)
            if fitted is None or fitted[1] is None or not fitted[1].valid:
                continue
            hypotheses.append((fitted[0], fitted[1]))

        if not hypotheses:
            return None

        hypotheses.sort(key=lambda h: h[1].residual_rms)
        best = hypotheses[0]

        # A symmetric waymark layout makes this genuinely undecidable from one frame. The
        # standard ring puts A/B/C/D on the cardinals and 1/2/3/4 on the intercardinals,
        # so mapping the lettered marks onto the numbered ones is the *same shape rotated
        # 45 degrees* — it fits exactly, and the residual cannot tell the two apart.
        #
        # Physics can. The character did not teleport and the camera did not spin 45
        # degrees between frames, so the previous estimate breaks the tie.
        runner_up = hypotheses[1] if len(hypotheses) > 1 else None
        self.last_was_ambiguous = bool(
            runner_up is not None
            and (runner_up[1].residual_rms - best[1].residual_rms) < self.ambiguity_tolerance_px
        )
        if self.last_was_ambiguous:
            resolved = self._break_tie_with_history(hypotheses)
            if resolved is not None:
                return resolved

        return best

    def _break_tie_with_history(
        self, hypotheses: list[tuple[list[Sighting], Similarity2D]]
    ) -> tuple[list[Sighting], Similarity2D] | None:
        """Pick the hypothesis most consistent with where we just were.

        Returns `None` when there is no usable history, which leaves the caller to report
        reduced confidence rather than commit to a coin flip.
        """
        previous = self.last
        if not previous.usable or previous.player is None:
            return None

        best: tuple[float, tuple[list[Sighting], Similarity2D]] | None = None
        for candidate in hypotheses:
            fit = candidate[1]
            player = fit.unapply(ArenaPoint(0.0, 0.0))
            yaw = normalise_deg(fit.rotation_deg)
            jump_m = player.distance_to(previous.player)
            spin_deg = abs((yaw - previous.pose.yaw_deg + 180.0) % 360.0 - 180.0)
            # Metres and degrees are not comparable, so weight the spin into roughly
            # metre-equivalents: a 45-degree jump is as implausible as several metres.
            score = jump_m + spin_deg * 0.15
            if best is None or score < best[0]:
                best = (score, candidate)

        return best[1] if best else None

    def _fit_assignment(
        self, sightings: list[Sighting]
    ) -> tuple[list[Sighting], Similarity2D | None] | None:
        usable = [s for s in sightings if self.layout.get(s.mark_id) is not None]
        if len(usable) < self.min_sightings:
            return None
        arena_points = [self.layout.get(s.mark_id) for s in usable]
        # Flip screen +y-down into a right-handed frame so the fitted rotation comes out
        # directly as a compass bearing. Without this the yaw is mirrored, which produces
        # a player that dodges the wrong way every single time.
        observed = [ArenaPoint(s.minimap_offset[0], -s.minimap_offset[1]) for s in usable]
        return usable, fit_similarity(arena_points, observed)

    # -- view path ---------------------------------------------------------------

    def from_view(
        self,
        sightings: list[Sighting],
        fallback: LocalizationResult | None = None,
        at: float | None = None,
    ) -> LocalizationResult:
        """Fit a ground-plane homography from waymarks visible in the 3D view.

        Needs four. Below that the projective fit is underdetermined and the right answer
        is to say so and let the minimap path carry — not to drop to an affine
        approximation, which looks fine near the centre of the screen and is badly wrong
        exactly at the edges where the off-screen mechanics live.
        """
        at = now() if at is None else at
        usable = [
            s
            for s in sightings
            if s.view_point is not None and self.layout.get(s.mark_id) is not None
        ]
        if len(usable) < 4:
            return self._fail(f"homography needs 4 marks in view, saw {len(usable)}", at)

        arena_points = [self.layout.get(s.mark_id) for s in usable]
        screen_points = [s.view_point for s in usable]

        homography = fit_homography(arena_points, screen_points)
        if homography is None or not math.isfinite(homography.residual_rms):
            return self._fail("homography fit failed", at)

        # The homography alone does not say where the player is standing — it maps the
        # ground plane, not the character. Position comes from the minimap; the homography
        # supplies projection and a cross-check on yaw.
        player = fallback.player if fallback and fallback.player else None
        if player is None:
            return self._fail("homography solved but no player position to anchor it", at)

        yaw = homography.yaw_at(player)
        if yaw is None:
            return self._fail("homography degenerate at player position", at)

        confidence = _confidence(homography.residual_rms, 25.0, len(usable))
        self.solves += 1
        self.last = LocalizationResult(
            player=player,
            pose=CameraPose(yaw_deg=normalise_deg(yaw), confidence=confidence),
            scale_px_per_m=fallback.scale_px_per_m if fallback else 0.0,
            homography=homography,
            confidence=confidence,
            method="homography",
            residual_px=homography.residual_rms,
            sightings_used=len(usable),
            at=at,
        )
        return self.last

    # -- combined ----------------------------------------------------------------

    def solve(self, sightings: list[Sighting], at: float | None = None) -> LocalizationResult:
        """Best available estimate. Minimap for position, homography for projection."""
        minimap = self.from_minimap(sightings, at)
        if not minimap.usable:
            return minimap

        view = self.from_view(sightings, fallback=minimap, at=at)
        if not view.usable:
            return minimap

        # Both solved. Keep the minimap's position and the homography's projection, and
        # use the disagreement in yaw as a health signal rather than averaging it away.
        disagreement = abs(
            (view.pose.yaw_deg - minimap.pose.yaw_deg + 180.0) % 360.0 - 180.0
        )
        if disagreement > 15.0:
            minimap.reason = f"homography yaw disagrees by {disagreement:.0f}deg; using minimap"
            return minimap

        combined = LocalizationResult(
            player=minimap.player,
            pose=view.pose,
            scale_px_per_m=minimap.scale_px_per_m,
            homography=view.homography,
            confidence=min(minimap.confidence, view.confidence),
            method="minimap+homography",
            residual_px=minimap.residual_px,
            sightings_used=max(minimap.sightings_used, view.sightings_used),
            at=at or now(),
        )
        self.last = combined
        return combined

    def _fail(self, reason: str, at: float) -> LocalizationResult:
        result = LocalizationResult(method="none", reason=reason, at=at)
        self.last = result
        return result

    def stats(self) -> dict[str, object]:
        return {
            "arena": self.layout.arena_id,
            "solves": self.solves,
            "rejections": self.rejections,
            "last": self.last.method,
            "reason": self.last.reason,
        }


def _injective_assignments(
    members: list[Sighting], marks: list[str]
) -> list[list[Sighting]]:
    """Every way to label these blobs with these marks, one mark each.

    When there are more blobs than marks — two red dots but the encounter only placed A —
    the surplus blobs are *dropped* rather than force-fitted. One of them is something
    else (a party dot bleeding through, a map decoration), and there is no way to tell
    which from colour alone, so every choice of which to keep becomes its own hypothesis
    and the residual picks.
    """
    keep = min(len(members), len(marks))
    if keep == 0:
        return []
    out: list[list[Sighting]] = []
    for subset in itertools.combinations(range(len(members)), keep):
        for labels in itertools.permutations(marks, keep):
            out.append([members[i].identified_as(m) for i, m in zip(subset, labels)])
    return out


def _confidence(residual: float, tolerance: float, count: int) -> float:
    """Confidence from fit quality and redundancy.

    Two marks give an exact fit, so a zero residual there means nothing — there was no
    freedom to be wrong. Redundancy is what makes a low residual meaningful, so the
    two-mark case is capped below the redundant case no matter how clean it looks.
    """
    quality = max(0.0, 1.0 - residual / max(tolerance, 1e-6))
    redundancy_cap = 0.75 if count <= 2 else 1.0
    return round(min(quality, redundancy_cap), 4)
