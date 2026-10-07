"""S04/S06: ordinary city errands, both household interiors and independent endings."""
from .city_life import workshop, verify_results, record_baseline, rejoin

STORIES = [
 ("life_welcome_seats","warden_petra","inn_guest_lio","city_welcome_seats"),
 ("life_welcome_shade","inn_guest_lio","guide_nola","city_welcome_shade"),
 ("life_welcome_signs","guide_nola","recruit_sela","city_welcome_signs"),
 ("life_welcome_rota","recruit_sela","guide_nola","city_welcome_rota"),
 ("life_welcome_corner","guide_nola","warden_petra",None),
 ("life_yard_paths","laundress_mira","laundry_kea","city_yard_paths"),
 ("life_yard_wind","laundry_kea","weaver_nessa","city_yard_wind"),
 ("life_yard_weather","weaver_nessa","lyra_tillerson","city_yard_weather"),
 ("life_yard_clear","lyra_tillerson","laundry_kea","city_yard_clear"),
 ("life_yard_agreement","laundry_kea","laundress_mira",None),
]
STATIONS = {
 "city_welcome_seats":(137,132,["feet","rise"]),
 "city_welcome_shade":(139,135,["path","shade"]),
 "city_welcome_signs":(204,106,["pictures","turn"]),
 "city_welcome_rota":(190,124,["hours","between"]),
 "city_yard_paths":(210,133,["separate","passage"]),
 "city_yard_wind":(197,131,["hem","air"]),
 "city_yard_weather":(201,130,["sun","rain"]),
 "city_yard_clear":(206,131,["bundles","rack","basin"]),
}
OUTCOMES={"life_welcome_corner":("shared_table","quiet_corners"),"life_yard_agreement":("marked_zones","folding_turns")}
HOMES={"welcome_house":((191.5,120.5),(191.5,122.5),(192.5,126.5)),
       "laundry_home":((212.5,138.5),(212.5,136.5),(212.5,134.5))}

def visit_home(j, name):
    from .journey import JourneyFailed
    outside, doorway, inside = HOMES[name]
    for x,z in (outside,doorway,inside):
        if not j.walk_to(x,z,y=74,radius=.65,label=name+" enter").ok:
            raise JourneyFailed("could not enter "+name)
    j.settle()
    before=j.me().get("server") or j.me()
    rejoin(j)
    after=j.me().get("server") or j.me()
    if abs(after["x"]-before["x"])>.8 or abs(after["z"]-before["z"])>.8 or abs(after["y"]-74)>.15:
        raise JourneyFailed("home position not preserved on rejoin: "+str(after))
    for x,z in (doorway,outside):
        if not j.walk_to(x,z,y=74,radius=.65,label=name+" exit").ok:
            raise JourneyFailed("could not leave "+name)
    return "walked through doorway, rejoined inside at y=74, walked out"

def steps(j, welcome=0, yard=0, yard_first=False):
    result=[]
    for q,giver,destination in [
        ("petra_welcome","warden_petra","tethis"),
        ("city_daily_bread","warden_petra","baker_elin"),
        ("city_harbor_supplies","baker_elin","dockhand_bram"),
        ("city_evening_table","dockhand_bram","inn_guest_lio"),
        ("city_homeward","inn_guest_lio","warden_petra"),
        ("city_clean_linen","healer_iona","weaver_nessa"),
        ("city_shared_water","weaver_nessa","laundress_mira"),
    ]:
        result += [("quest","accept "+q,lambda q=q,n=giver:j.quest(n,q,"accept")),
                   ("quest","complete "+q,lambda q=q,n=destination:j.quest(n,q,"complete"))]
    baseline={}
    result.append(("verify","record earned-coin baseline",lambda:record_baseline(j,baseline)))
    for name in HOMES:
        result.append(("walk","enter, save and leave "+name,lambda n=name:visit_home(j,n)))
    outcomes={"life_welcome_corner":OUTCOMES["life_welcome_corner"][welcome],"life_yard_agreement":OUTCOMES["life_yard_agreement"][yard]}
    ordered=STORIES[5:]+STORIES[:5] if yard_first else STORIES
    for q,giver,destination,station in ordered:
        result.append(("quest","accept "+q,lambda q=q,n=giver:j.quest(n,q,"accept")))
        if station:
            resume="city_welcome_rota" if station.startswith("city_welcome_") else "city_yard_paths"
            result.append(("workshop",station,lambda q=q,s=station,r=resume:workshop(j,q,s,stations=STATIONS,resume_station=r)))
        action="complete:"+outcomes[q] if q in outcomes else "complete"
        result.append(("quest","complete "+q,lambda q=q,n=destination,a=action:j.quest(n,q,a)))
    for npc,quest in [("guide_nola","life_welcome_corner"),("laundry_kea","life_yard_agreement")]:
        result.append(("verify","rejoin and revisit "+npc,lambda n=npc,q=quest:verify_results(j,outcomes,baseline,stories=STORIES,reward=100,revisit_npc=n,revisit_quest=q)))
    return result
