"""Luminara city-life acceptance through ordinary player controls; no setup cheats."""
from __future__ import annotations
import time
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .journey import Journey

# Authored test expectations, deliberately independent of the game's runtime registry.
STORIES = [
 ("life_bread_samples", "baker_elin", "baker_elin", "city_bread_board"),
 ("life_bread_shift", "baker_elin", "dockhand_bram", None),
 ("life_bread_herbs", "dockhand_bram", "lyra_tillerson", None),
 ("life_bread_methods", "lyra_tillerson", "oven_pella", "city_bread_trial"),
 ("life_bread_table", "oven_pella", "baker_elin", None),
 ("life_cloth_patterns", "weaver_nessa", "weaver_sister", "city_cloth_frame"),
 ("life_cloth_wash", "weaver_sister", "laundress_mira", "city_cloth_wash"),
 ("life_cloth_agreement", "laundress_mira", "weaver_nessa", None),
 ("life_cloth_window", "weaver_nessa", "innkeeper_sella", "city_window_display"),
 ("life_water_marks", "laundress_mira", "waterwright_teren", "city_old_basin"),
 ("life_water_memory", "waterwright_teren", "miller_ada", None),
 ("life_water_witness", "miller_ada", "elder_esme", None),
 ("life_water_trial", "elder_esme", "miller_ada", "city_mill_trial"),
 ("life_water_contract", "miller_ada", "maren_goleli", "city_water_archive"),
 ("life_water_future", "maren_goleli", "waterwright_teren", None),
]
STATIONS = {
 "city_bread_board": (91,129, ["cut","separate"]),
 "city_bread_trial": (95,129, ["wrap","remove","limit"]),
 "city_cloth_frame": (100,129, ["compare","privacy"]),
 "city_cloth_wash": (206,132, ["basin","hem"]),
 "city_window_display": (196,130, ["hang","turn"]),
 "city_old_basin": (211,138, ["wear","limits"]),
 "city_mill_trial": (91,85, ["fit","dates"]),
 "city_water_archive": (110,131, ["source","combine"]),
}
OUTCOMES = {
 "life_bread_table": ("wrapped","reheated"),
 "life_cloth_window": ("layered","folded"),
 "life_water_future": ("repair","exhibit"),
}

def rejoin(j: Journey) -> None:
    j.stop()
    j.drain_events()
    j.gw.call("enter", timeout_s=40, zone="overworld")
    j.wait_event("quest_update", timeout_s=20)
    time.sleep(1)
    j.drain_events()

def workshop(j: Journey, quest_id: str, station: str, *, stations=STATIONS, feet_y=74, resume_station="city_mill_trial") -> str:
    from .journey import JourneyFailed
    x,z,answers = stations[station]
    import math
    for attempt in range(3):
        leg = j.walk_to(x + .5, z + .5, y=feet_y, radius=1.4, label="workshop " + station + (f" (settle {attempt})" if attempt else ""))
        if not leg.ok:
            raise JourneyFailed("could not reach workshop " + station)
        j.settle()
        actual = j.me().get("server") or j.me()
        if math.hypot(actual["x"] - x - .5, actual["z"] - z - .5) <= 1.65 and abs(actual["y"] - feet_y) <= 2:
            break
    else:
        raise JourneyFailed(f"did not settle within workshop range: {actual}")
    def open_node():
        j.drain_events()
        j.gw.call("landmark", landmarkId=station)
        return j.wait_event("npc_dialog", lambda d: str(d.get("nodeId","")).startswith("life:" + station + ":"))
    node = open_node()
    for index, answer in enumerate(answers):
        if node["nodeId"] != f"life:{station}:{index}":
            raise JourneyFailed(f"wrong step: {node}")
        # A wrong observation must keep the same step and allow a corrected answer.
        j.gw.call("choose", entityId=node["entityId"], nodeId=node["nodeId"], optionId="reconsider")
        retry = j.wait_event("npc_dialog", lambda d: str(d.get("messageKey","")).startswith(f"life.{station}.feedback."))
        if retry["nodeId"] != node["nodeId"]:
            raise JourneyFailed("wrong observation advanced the workshop")
        j.gw.call("choose", entityId=node["entityId"], nodeId=node["nodeId"], optionId=answer)
        node = j.wait_event("npc_dialog", lambda d: d.get("nodeId") == f"life:{station}:{index+1}")
        if station == resume_station and index == 0:
            j.gw.fire("close_dialog")
            rejoin(j)
            if (j.quest_entry(quest_id) or {}).get("have") != 1:
                raise JourneyFailed("partial workshop did not survive rejoin")
            node = open_node()
    j.gw.fire("close_dialog")
    j.gw.wait_snapshot(lambda d: any(q.get("id")==quest_id and q.get("state")=="ready" for q in d.get("quests",[])), timeout_s=8, label="workshop ready")
    return f"{len(answers)} steps, retry checked" + ("; partial rejoin checked" if station==resume_station else "")

def verify_results(j: Journey, outcomes: dict[str,str], baseline: dict, *, stories=STORIES, reward=150, revisit_npc="waterwright_teren", revisit_quest="life_water_future") -> str:
    from .journey import JourneyFailed
    rejoin(j)
    for q,_,_,_ in stories:
        if j.quest_state(q) != "completed":
            raise JourneyFailed(q + " not completed after rejoin")
    for q,c in outcomes.items():
        if (j.quest_entry(q) or {}).get("choiceId") != c:
            raise JourneyFailed(q + " choice lost after rejoin")
    if j.me()["coins"] - baseline["coins"] != reward:
        raise JourneyFailed(f"expected exactly {reward} story coins; baseline={baseline['coins']}, actual={j.me()['coins']}")
    if j.quest_state("story_manifest") not in (None, "available"):
        raise JourneyFailed("optional city stories advanced the main story")
    npc = j.approach_npc(revisit_npc)
    # Revisit through the ordinary topic menu; different players can advance idle lines.
    found = False
    for _ in range(4):
        j.drain_events()
        j.gw.call("talk", entityId=npc["entityId"])
        msg = j.wait_event("npc_dialog")
        if msg.get("nodeId") == "quest_menu":
            if any(o.get("id")==revisit_quest for o in msg.get("options",[])):
                raise JourneyFailed("completed quest offered for a repeat claim")
            j.gw.call("choose", entityId=npc["entityId"], nodeId=msg["nodeId"], optionId="talk")
            msg = j.wait_event("npc_dialog")
        found |= msg.get("messageKey") == f"quest.{revisit_quest}.revisit.{outcomes[revisit_quest]}.{revisit_npc}"
        j.gw.fire("close_dialog")
        if found: break
    if not found:
        raise JourneyFailed("saved result missing from resident revisit")
    if j.me()["coins"] - baseline["coins"] != reward:
        raise JourneyFailed("revisit changed coins")
    return f"{len(stories)} quests; {len(outcomes)} durable outcomes; {reward} coins once; main story unchanged; resident revisit"

def record_baseline(j: Journey, baseline: dict) -> str:
    # quest_update is an event; latest() can still be the previous periodic snapshot.
    before = j.gw.latest().seq
    j.gw.fire("snapshot")
    deadline = time.monotonic() + 5
    while j.gw.latest().seq <= before and time.monotonic() < deadline:
        time.sleep(.02)
    if j.gw.latest().seq <= before:
        from .journey import JourneyFailed
        raise JourneyFailed("no fresh reward snapshot")
    baseline["coins"] = j.me()["coins"]
    return f"baseline {baseline['coins']} coins after prerequisites"


def steps(j: Journey, alternate: bool = False):
    result = []
    def visit(q,giver,destination):
        result.extend([
            ("quest", "accept " + q, lambda q=q,n=giver: j.quest(n,q,"accept")),
            ("quest", "complete " + q, lambda q=q,n=destination: j.quest(n,q,"complete")),
        ])
    visit("petra_welcome","warden_petra","tethis")
    visit("city_daily_bread","warden_petra","baker_elin")
    visit("city_clean_linen","healer_iona","weaver_nessa")
    baseline = {}
    result.append(("verify", "record earned-coin baseline", lambda: record_baseline(j, baseline)))
    outcomes = {q: choices[int(alternate)] for q,choices in OUTCOMES.items()}
    for q,giver,destination,station in STORIES:
        result.append(("quest","accept " + q,lambda q=q,n=giver: j.quest(n,q,"accept")))
        if station:
            result.append(("workshop",station,lambda q=q,s=station: workshop(j,q,s)))
        action = "complete:" + outcomes[q] if q in outcomes else "complete"
        result.append(("quest","complete " + q,lambda q=q,n=destination,a=action: j.quest(n,q,a)))
    result.append(("verify","rejoin and revisit",lambda: verify_results(j,outcomes,baseline)))
    return result

