"""S14 acceptance: fresh arrival walks into Duskhollow; ordinary controls only."""
from .city_life import record_baseline, workshop, verify_results

STORIES = [
    ("life_fungi_samples", "ember_innkeep", "grower_siv", "hollow_fungi_samples"),
    ("life_fungi_bed", "grower_siv", "grower_siv", "hollow_fungi_bed"),
    ("life_fungi_pot", "grower_siv", "cook_bren", "hollow_fungi_pot"),
    ("life_fungi_opaline", "cook_bren", "opaline_vess", None),
    ("life_fungi_carrier", "opaline_vess", "carrier_ona", None),
    ("life_fungi_household", "carrier_ona", "nightworker_rusk", None),
    ("life_fungi_menu", "nightworker_rusk", "ember_innkeep", None),
]
STATIONS = {
    "hollow_fungi_samples": (44, 19, ["labels", "texture"]),
    "hollow_fungi_bed": (47, 17, ["moisture", "shade", "spacing"]),
    "hollow_fungi_pot": (44, 30, ["stems", "number"]),
}

def arrival_steps(j, baseline):
    from .journey import ROUTES, JourneyFailed
    def descend():
        for x,z,y in [(58.5,100.5,64), (58.5,56,25), (58,46,25), (50,36,25)]:
            leg=j.walk_to(x,z,y=y,radius=1.0 if y != 25 else 2.0,label="Delver's Descent")
            if not leg.ok: raise JourneyFailed("could not descend the mine stairs")
        if j.me()["y"] > 28: raise JourneyFailed("not on the cavern floor")
        return "walked from the surface into Duskhollow at y=25"
    return [
        ("verify", "record fresh character coins", lambda: record_baseline(j,baseline)),
        ("walk", "west causeway to minehead", lambda: j.walk_route([(128,116)] + ROUTES["city_to_ironvein"] + [(60,108),(58,104)], "minehead")),
        ("walk", "descend the complete mine stairs", descend),
    ]


def steps(j, alternate=False):
    baseline = {}
    outcomes = {"life_fungi_menu": "rotating_pots" if alternate else "daily_special"}
    result = arrival_steps(j, baseline)
    for q,giver,destination,station in STORIES:
        result.append(("quest", "accept " + q, lambda q=q,n=giver: j.quest(n,q,"accept")))
        if station:
            result.append(("workshop", station, lambda q=q,s=station: workshop(j,q,s,stations=STATIONS,feet_y=25,resume_station="hollow_fungi_bed")))
        action = "complete:" + outcomes[q] if q in outcomes else "complete"
        result.append(("quest", "complete " + q, lambda q=q,n=destination,a=action: j.quest(n,q,a)))
    result.append(("verify", "saved outcome and personal revisit", lambda: verify_results(j,outcomes,baseline,stories=STORIES,reward=70,revisit_npc="ember_innkeep",revisit_quest="life_fungi_menu")))
    return result
