"""S34 and S14+S34: fresh players, real mine stairs, no setup commands."""
from .city_life import record_baseline, rejoin, workshop, verify_results
from .duskhollow_life import arrival_steps, STORIES as FUNGI_STORIES, STATIONS as FUNGI_STATIONS

STORIES = [
    ("life_vent_model", "ventwright_marda", "ventwright_marda", "hollow_vent_model"),
    ("life_vent_memory", "ventwright_marda", "carer_elis", "hollow_vent_memory"),
    ("life_vent_repairs", "carer_elis", "grenn_stonewall", "hollow_vent_repairs"),
    ("life_vent_records", "grenn_stonewall", "brondt_ashvein", "hollow_vent_records"),
    ("life_vent_future", "brondt_ashvein", "ventwright_marda", None),
]
STATIONS = {
    "hollow_vent_model": (57, 16, ["direction", "baffle"]),
    "hollow_vent_memory": (63, 17, ["pause", "limits"]),
    "hollow_vent_repairs": (69, 19, ["overlap"]),
    "hollow_vent_records": (63, 13, ["sequence", "contributors"]),
}


def demonstrate(j):
    """Repeat both baffles via current dialog options, then reconnect unchanged."""
    import copy
    from .journey import JourneyFailed
    leg = j.walk_to(57.5, 16.5, y=25, radius=1.3, label="return to ventilation model")
    if not leg.ok:
        raise JourneyFailed("could not return to ventilation model")
    j.settle()
    baseline = {}
    record_baseline(j, baseline)
    saved = {q["id"]: copy.deepcopy(j.quest_entry(q["id"])) for q in j.snap().get("quests", []) if q["id"].startswith(("life_fungi_", "life_vent_"))}
    for _ in range(2):
        j.drain_events()
        j.gw.call("landmark", landmarkId="hollow_vent_model")
        node = j.wait_event("npc_dialog", lambda d: str(d.get("nodeId", "")).startswith("life:hollow_vent_model:demo:"))
        for option in ("old_baffle", "revised_baffle"):
            j.gw.call("choose", entityId=node["entityId"], nodeId=node["nodeId"], optionId=option)
            node = j.wait_event("npc_dialog", lambda d, o=option: d.get("messageKey") == "life.hollow_vent_model.demo.result." + o)
        j.gw.fire("close_dialog")
    rejoin(j)
    after = {}
    record_baseline(j, after)
    if after != baseline or any(j.quest_entry(q) != entry for q, entry in saved.items()):
        raise JourneyFailed("repeating the model changed quests or coins")
    return "both baffles repeated twice; close/reopen and rejoin; quest journal and coins unchanged"


def steps(j, alternate=False, fungi=None, vent_first=False):
    baseline = {}
    outcomes = {"life_vent_future": "travel_display" if alternate else "teaching_wall"}
    stories = list(STORIES)
    stations = dict(STATIONS)
    if fungi is not None:
        outcomes["life_fungi_menu"] = "rotating_pots" if fungi else "daily_special"
        stories = stories + FUNGI_STORIES if vent_first else FUNGI_STORIES + stories
        stations.update(FUNGI_STATIONS)
    result = arrival_steps(j, baseline)
    for q, giver, destination, station in stories:
        result.append(("quest", "accept " + q, lambda q=q, n=giver: j.quest(n, q, "accept")))
        if station:
            resume = station if station in ("hollow_fungi_bed", "hollow_vent_model") else ""
            result.append(("workshop", station, lambda q=q, s=station, r=resume: workshop(j, q, s, stations=stations, feet_y=25, resume_station=r)))
        action = "complete:" + outcomes[q] if q in outcomes else "complete"
        result.append(("quest", "complete " + q, lambda q=q, n=destination, a=action: j.quest(n, q, a)))
    result.append(("verify", "saved outcomes and ventilation revisit", lambda: verify_results(j, outcomes, baseline, stories=stories, reward=50 if fungi is None else 120, revisit_npc="ventwright_marda", revisit_quest="life_vent_future")))
    result.append(("verify", "repeat ventilation demonstration without reward", lambda: demonstrate(j)))
    return result
