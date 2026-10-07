"""Score a FantCraft playtest: this player's perception and decisions against
the game's own ground truth.

Three logs describe one run from three vantage points:

* the **oracle session** (FantCraft `var/playtest/<id>/`): `truth.jsonl` is
  what the server believed at 4Hz, `client.jsonl` what the browser rendered,
  `events.jsonl` every skill press outcome — including the rejections the
  player never sees on screen;
* the **player session** (`sessions/<id>/`): what this player perceived
  (`state` trace rows) and pressed (`dispatch` rows).

The join is wall-clock: oracle rows carry Date.now() ms, player rows carry a
monotonic offset anchored by `started_unix` in the session meta. Nearest-
neighbour within a tolerance, because neither side samples on the other's
schedule.

The output is a playtest report: perception error per field, press outcome
histogram, and a findings list — the findings are the point, they are what
feeds back into FantCraft's backlog.

    python -m tools.fancraft_report \
        --oracle D:/projects/fancraft/var/playtest/<id> \
        --session sessions/<id> \
        [--json report.json]

Run it with only --oracle to analyse a human (or bridge-only) session: press
outcomes and truth-vs-client lag still work without a player session.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# ── loading ──────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn tail line from a killed process is expected


@dataclass(slots=True)
class OracleSession:
    root: Path
    truth: list[dict[str, Any]] = field(default_factory=list)
    client: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] | None = None

    @classmethod
    def load(cls, root: Path) -> "OracleSession":
        s = cls(root)
        s.truth = sorted(read_jsonl(root / "truth.jsonl"), key=lambda r: r.get("t", 0))
        s.client = sorted(read_jsonl(root / "client.jsonl"), key=lambda r: r.get("t", 0))
        s.events = sorted(read_jsonl(root / "events.jsonl"), key=lambda r: r.get("t", 0))
        mpath = root / "manifest.json"
        if mpath.exists():
            s.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        return s


@dataclass(slots=True)
class PlayerSession:
    root: Path
    # (wall_ms, fields) per perceived state
    states: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    # (wall_ms, key, action) per dispatched input
    dispatches: list[tuple[float, str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "PlayerSession":
        s = cls(root)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        anchor = meta.get("started_unix")
        if anchor is None:
            # Older session without the precise anchor: fall back to the
            # 1s-resolution wall stamp and warn — joins get a coarse spine.
            anchor = datetime.fromisoformat(meta["started_wall"]).timestamp()
            print("[report] warning: session lacks started_unix; join is +/-1s", file=sys.stderr)
        anchor_ms = float(anchor) * 1000.0

        for row in read_jsonl(root / "trace.jsonl"):
            t_ms = anchor_ms + float(row.get("t", 0.0)) * 1000.0
            kind = row.get("kind")
            data = row.get("data", {})
            if kind == "state":
                s.states.append((t_ms, data.get("fields", {})))
            elif kind == "dispatch":
                s.dispatches.append((t_ms, str(data.get("key", "")), str(data.get("action", ""))))
        return s


# ── joining ──────────────────────────────────────────────────────────


class TimeSeries:
    """Nearest-neighbour lookup over (t_ms, row) pairs."""

    def __init__(self, rows: list[dict[str, Any]], key: str = "t") -> None:
        self.rows = rows
        self.times = [float(r.get(key, 0)) for r in rows]

    def nearest(self, t_ms: float, tolerance_ms: float) -> dict[str, Any] | None:
        if not self.rows:
            return None
        i = bisect_left(self.times, t_ms)
        best: dict[str, Any] | None = None
        best_dt = tolerance_ms
        for j in (i - 1, i):
            if 0 <= j < len(self.rows):
                dt = abs(self.times[j] - t_ms)
                if dt <= best_dt:
                    best, best_dt = self.rows[j], dt
        return best


def summarize_errors(errors: list[float]) -> dict[str, float] | None:
    if not errors:
        return None
    ordered = sorted(errors)
    return {
        "n": len(errors),
        "mean": round(statistics.fmean(errors), 4),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "max": round(ordered[-1], 4),
    }


# ── analyses ─────────────────────────────────────────────────────────


def perception_accuracy(
    oracle: OracleSession, player: PlayerSession, tolerance_ms: float
) -> dict[str, Any]:
    """Player-perceived fractions vs server truth, matched by wall clock."""
    truth = TimeSeries(oracle.truth)
    pairs = {"player.hp_frac": "hpFrac", "player.mp_frac": "mpFrac"}
    out: dict[str, Any] = {}
    for player_field, truth_field in pairs.items():
        errors: list[float] = []
        matched = 0
        for t_ms, fields in player.states:
            seen = fields.get(player_field)
            if not isinstance(seen, (int, float)):
                continue
            row = truth.nearest(t_ms, tolerance_ms)
            if row is None or truth_field not in row:
                continue
            matched += 1
            errors.append(abs(float(seen) - float(row[truth_field])))
        stats = summarize_errors(errors)
        out[player_field] = {"matched": matched, **(stats or {})} if stats else {"matched": 0}
    return out


def indicator_agreement(
    oracle: OracleSession, player: PlayerSession, tolerance_ms: float
) -> dict[str, Any]:
    """Player-perceived HUD indicators vs what the client says it rendered.

    The client's `slotFlags` are DOM truth for the range wash, the combo ring,
    and the cooldown state; the player's `action.<id>.*` fields are what the
    probes read off the pixels. Disagreement here is pure perception error —
    the render hop is already accounted for.
    """
    slots = (oracle.manifest or {}).get("hotbarSlots") or []
    ability_by_index = [s.get("abilityId") for s in slots]
    if not any(ability_by_index):
        return {}

    client = TimeSeries(oracle.client)
    pairs = {
        "out_of_range": "outOfRange",
        "combo_next": "comboNext",
        "ready": "onCooldown",  # inverted below
    }
    counts = {k: {"n": 0, "agree": 0} for k in pairs}
    for t_ms, fields in player.states:
        row = client.nearest(t_ms, tolerance_ms)
        if row is None:
            continue
        flags = row.get("slotFlags") or []
        for idx, ability in enumerate(ability_by_index):
            if not ability or idx >= len(flags):
                continue
            for field_suffix, flag_name in pairs.items():
                seen = fields.get(f"action.{ability}.{field_suffix}")
                if not isinstance(seen, bool):
                    continue
                truth = bool(flags[idx].get(flag_name))
                if field_suffix == "ready":
                    truth = not truth  # ready is the absence of on-cooldown
                counts[field_suffix]["n"] += 1
                counts[field_suffix]["agree"] += int(seen == truth)
    return {
        k: {"n": v["n"], "agreement": round(v["agree"] / v["n"], 3) if v["n"] else None}
        for k, v in counts.items()
        if v["n"]
    }


def render_lag(oracle: OracleSession, tolerance_ms: float) -> dict[str, Any]:
    """Server truth vs what the client rendered — the hop perception can't fix."""
    truth = TimeSeries(oracle.truth)
    errors: list[float] = []
    for row in oracle.client:
        hp = row.get("hp")
        if not isinstance(hp, dict) or not hp.get("max"):
            continue
        t = truth.nearest(float(row.get("t", 0)), tolerance_ms)
        if t is None or not t.get("maxHp"):
            continue
        errors.append(abs(hp["cur"] / hp["max"] - float(t["hpFrac"])))
    stats = summarize_errors(errors)
    return {"hp_frac_client_vs_truth": stats or {"n": 0}}


def press_outcomes(oracle: OracleSession) -> dict[str, Any]:
    """Histogram of skill_use outcomes: what the server did with every press."""
    by_outcome: dict[str, int] = {}
    by_skill: dict[str, dict[str, int]] = {}
    for ev in oracle.events:
        if ev.get("kind") != "skill_use":
            continue
        outcome = str(ev.get("outcome", "?"))
        skill = str(ev.get("skillId", "?"))
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        by_skill.setdefault(skill, {})[outcome] = by_skill.setdefault(skill, {}).get(outcome, 0) + 1
    total = sum(by_outcome.values())
    ok = by_outcome.get("ok", 0)
    return {
        "total_presses": total,
        "ok": ok,
        "efficiency": round(ok / total, 3) if total else None,
        "by_outcome": dict(sorted(by_outcome.items(), key=lambda kv: -kv[1])),
        "by_skill": by_skill,
    }


def deaths(oracle: OracleSession) -> int:
    return sum(1 for ev in oracle.events if ev.get("kind") == "player_death")


def build_findings(report: dict[str, Any]) -> list[str]:
    """Turn the numbers into the sentences a developer acts on.

    Thresholds are opinions, stated inline; the raw numbers travel alongside so
    a reader can disagree with the opinion without re-running anything.
    """
    findings: list[str] = []
    presses = report.get("presses", {})
    by_outcome = presses.get("by_outcome", {})
    total = presses.get("total_presses", 0) or 0

    def share(outcome: str) -> float:
        return by_outcome.get(outcome, 0) / total if total else 0.0

    if total and presses.get("efficiency", 1) is not None and presses["efficiency"] < 0.7:
        findings.append(
            f"Press efficiency is {presses['efficiency']:.0%} — under 70%. The outcome "
            "histogram below says where the losses are."
        )
    if share("range") > 0.10:
        findings.append(
            f"{share('range'):.0%} of presses failed on range. The HUD shows no "
            "range/distance indicator, so neither the tester nor a player can tell an "
            "in-range target from an out-of-range one before pressing. Consider a "
            "range tint on the hotbar (FF14 greys out out-of-range actions)."
        )
    if by_outcome.get("combo", 0) > 0:
        findings.append(
            f"{by_outcome['combo']} presses rejected as combo-broken. The HUD renders "
            "no combo-state indicator; the tester deliberately avoids combo abilities "
            "because they are unplayable from the screen alone (games/fancraft/combat.py). "
            "A glowing border on the next combo action would fix both bot and human."
        )
    if share("no_target") > 0.10:
        findings.append(
            f"{share('no_target'):.0%} of presses failed on no_target — target "
            "acquisition is lagging the rotation. Check the acquire_target reflex "
            "cadence vs the target-frame render delay."
        )
    if share("cooldown") > 0.15:
        findings.append(
            f"{share('cooldown'):.0%} of presses failed on cooldown — the tester is "
            "pressing into a rolling GCD. Either the cooldown-overlay probe is "
            "misreading or the client's GCD display lags the server."
        )

    perception = report.get("perception", {})
    for fname, stats in perception.items():
        if stats.get("n") and stats.get("p95", 0) > 0.08:
            findings.append(
                f"Perception error on {fname}: p95 {stats['p95']:.3f} over {stats['n']} "
                "samples. Above 0.05 usually means probe misalignment (stale manifest, "
                "browser not fullscreen) rather than genuine lag."
            )
        if stats.get("matched", 0) == 0 and report.get("player_session"):
            findings.append(
                f"No joined samples for {fname} — clocks may be unanchored, or the "
                "player never perceived that field. Nothing was scored."
            )

    for name, stats in (report.get("indicators") or {}).items():
        if stats.get("n", 0) >= 20 and (stats.get("agreement") or 0) < 0.9:
            findings.append(
                f"Indicator perception weak: {name} agrees with the client render only "
                f"{stats['agreement']:.0%} of the time over {stats['n']} readings — "
                "probe threshold or region drift; recalibrate against fresh frames."
            )

    lag = report.get("render_lag", {}).get("hp_frac_client_vs_truth", {})
    if lag.get("n") and lag.get("p95", 0) > 0.08:
        findings.append(
            f"Client render lags server truth: HP p95 error {lag['p95']:.3f}. That gap "
            "is invisible to screen perception by definition — it is a networking or "
            "client-interpolation issue in the game, not a tester issue."
        )

    if report.get("deaths", 0) > 0:
        findings.append(f"Player died {report['deaths']} time(s) — see events.jsonl for context.")

    if not findings:
        findings.append("Nothing above thresholds. Raise the bar or lengthen the run.")
    return findings


# ── entry ────────────────────────────────────────────────────────────


def build_report(
    oracle_dir: Path, session_dir: Path | None, tolerance_ms: float
) -> dict[str, Any]:
    oracle = OracleSession.load(oracle_dir)
    report: dict[str, Any] = {
        "oracle_session": str(oracle_dir),
        "player_session": str(session_dir) if session_dir else None,
        "truth_samples": len(oracle.truth),
        "client_samples": len(oracle.client),
        "events": len(oracle.events),
        "manifest_loaded": oracle.manifest is not None,
        "presses": press_outcomes(oracle),
        "render_lag": render_lag(oracle, tolerance_ms),
        "deaths": deaths(oracle),
    }
    if session_dir is not None:
        player = PlayerSession.load(session_dir)
        report["player_states"] = len(player.states)
        report["player_dispatches"] = len(player.dispatches)
        report["perception"] = perception_accuracy(oracle, player, tolerance_ms)
        report["indicators"] = indicator_agreement(oracle, player, tolerance_ms)
    report["findings"] = build_findings(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# FantCraft playtest report", ""]
    lines.append(f"- oracle: `{report['oracle_session']}` "
                 f"({report['truth_samples']} truth / {report['client_samples']} client "
                 f"/ {report['events']} events)")
    if report.get("player_session"):
        lines.append(f"- player: `{report['player_session']}` "
                     f"({report.get('player_states', 0)} states, "
                     f"{report.get('player_dispatches', 0)} dispatches)")
    lines.append("")

    p = report["presses"]
    lines.append("## Presses")
    if p["total_presses"]:
        lines.append(f"- {p['total_presses']} presses, {p['ok']} resolved "
                     f"(efficiency {p['efficiency']:.0%})")
        for outcome, n in p["by_outcome"].items():
            lines.append(f"  - {outcome}: {n}")
    else:
        lines.append("- no skill presses recorded")
    lines.append("")

    if "perception" in report:
        lines.append("## Perception vs truth")
        for fname, stats in report["perception"].items():
            if stats.get("n"):
                lines.append(f"- {fname}: mean {stats['mean']:.3f}, p95 {stats['p95']:.3f}, "
                             f"max {stats['max']:.3f} over {stats['n']} samples")
            else:
                lines.append(f"- {fname}: no joined samples")
        lines.append("")

    if report.get("indicators"):
        lines.append("## HUD indicators vs client render")
        for name, stats in report["indicators"].items():
            lines.append(f"- {name}: {stats['agreement']:.1%} agreement over {stats['n']} joined readings")
        lines.append("")

    lag = report["render_lag"]["hp_frac_client_vs_truth"]
    lines.append("## Client render vs truth")
    if lag.get("n"):
        lines.append(f"- hp_frac: mean {lag['mean']:.3f}, p95 {lag['p95']:.3f} "
                     f"over {lag['n']} samples")
    else:
        lines.append("- no client samples to compare")
    lines.append("")

    lines.append("## Findings")
    for f in report["findings"]:
        lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--oracle", required=True, help="FantCraft var/playtest/<id> directory")
    parser.add_argument("--session", default=None, help="player_mk1 sessions/<id> directory")
    parser.add_argument("--tolerance-ms", type=float, default=600.0,
                        help="max clock distance for a truth/perception join")
    parser.add_argument("--json", default=None, help="also write the raw report as JSON")
    args = parser.parse_args(argv)

    oracle_dir = Path(args.oracle)
    if not oracle_dir.exists():
        print(f"[report] oracle directory not found: {oracle_dir}", file=sys.stderr)
        return 1
    session_dir = Path(args.session) if args.session else None
    if session_dir is not None and not session_dir.exists():
        print(f"[report] player session not found: {session_dir}", file=sys.stderr)
        return 1

    report = build_report(oracle_dir, session_dir, args.tolerance_ms)
    print(render_markdown(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[report] raw JSON -> {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
