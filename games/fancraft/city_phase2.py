"""Second Luminara slice, played from a fresh ordinary account."""
from .city_life import workshop, verify_results, record_baseline

STORIES = [
 ("life_cart_noise","dockhand_bram","apprentice_toma","city_cart_noise"),
 ("life_cart_wheel","apprentice_toma","apprentice_toma","city_cart_wheel"),
 ("life_cart_short","apprentice_toma","portmaster_dain","city_cart_short"),
 ("life_cart_detour","portmaster_dain","dockhand_bram","city_cart_detour"),
 ("life_cart_rest","dockhand_bram","dockhand_bram",None),
 ("life_supper_lio","innkeeper_sella","inn_guest_lio",None),
 ("life_supper_bram","inn_guest_lio","dockhand_bram",None),
 ("life_supper_orel","dockhand_bram","fisher_orel",None),
 ("life_supper_samples","fisher_orel","fisher_orel","city_supper_samples"),
 ("life_supper_table","fisher_orel","innkeeper_sella","city_supper_table"),
 ("life_supper_host","innkeeper_sella","inn_guest_lio",None),
]
STATIONS = {
 "city_cart_noise": (189,155,["surface","wheel","corners"]),
 "city_cart_wheel": (194,155,["washer","pads"]),
 "city_cart_short": (189,161,["clear"]),
 "city_cart_detour": (194,161,["turn","hours"]),
 "city_supper_samples": (171,104,["salt","labels"]),
 "city_supper_table": (175,108,["space","wash","introduce"]),
}
OUTCOMES = {"life_cart_rest": ("padded","rest_route"), "life_supper_host": ("shared_pot","two_pots")}

def steps(j, alternate=False):
    result = []
    for q,giver,destination in [
        ("petra_welcome","warden_petra","tethis"),
        ("city_daily_bread","warden_petra","baker_elin"),
        ("city_harbor_supplies","baker_elin","dockhand_bram"),
        ("city_evening_table","dockhand_bram","inn_guest_lio"),
    ]:
        result.extend([
          ("quest","accept "+q,lambda q=q,n=giver: j.quest(n,q,"accept")),
          ("quest","complete "+q,lambda q=q,n=destination: j.quest(n,q,"complete")),
        ])
    baseline = {}
    result.append(("verify","record earned-coin baseline",lambda: record_baseline(j, baseline)))
    outcomes = {q:c[int(alternate)] for q,c in OUTCOMES.items()}
    for q,giver,destination,station in STORIES:
        result.append(("quest","accept "+q,lambda q=q,n=giver: j.quest(n,q,"accept")))
        if station:
            result.append(("workshop",station,lambda q=q,s=station: workshop(j,q,s,stations=STATIONS,
                feet_y=67 if s.startswith("city_cart_") else 74,resume_station="city_cart_wheel")))
        action = "complete:"+outcomes[q] if q in outcomes else "complete"
        result.append(("quest","complete "+q,lambda q=q,n=destination,a=action: j.quest(n,q,a)))
    result.append(("verify","rejoin and revisit",lambda: verify_results(j,outcomes,baseline,
        stories=STORIES,reward=110,revisit_npc="inn_guest_lio",revisit_quest="life_supper_host")))
    return result
