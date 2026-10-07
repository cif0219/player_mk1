"""S18/S35: ordinary mountain journeys, workshop observations and repeatable lift trial."""
from .city_life import record_baseline, workshop, verify_results, rejoin

STORIES = [
 ("life_lift_timber","lift_talric","sawyer_bettany",None),
 ("life_lift_balance","sawyer_bettany","lift_talric","ridge_lift_balance"),
 ("life_lift_brake","lift_talric","lift_talric","ridge_lift_brake"),
 ("life_lift_signal","lift_talric","captain_renn","ridge_lift_signal"),
 ("life_lift_roles","captain_renn","lift_talric","ridge_lift_roles"),
 ("life_lift_plan","lift_talric","lift_talric",None),
 ("life_rope_knots","ropewright_ossa","ropewright_ossa","ridge_rope_knots"),
 ("life_rope_wear","ropewright_ossa","lensgrinder_daro","ridge_rope_wear"),
 ("life_rope_records","lensgrinder_daro","librarian_neris","ridge_rope_records"),
 ("life_rope_credit","librarian_neris","lensgrinder_daro","ridge_rope_credit"),
 ("life_rope_future","lensgrinder_daro","ropewright_ossa",None),
]
STATIONS = {
 "ridge_lift_balance":(-1,-144,["marks","balance"]),
 "ridge_lift_brake":(-5,-145,["brake","backup","buffer"]),
 "ridge_lift_signal":(12,-179,["sight","reply"]),
 "ridge_lift_roles":(0,-147,["swap","pause"]),
 "ridge_rope_knots":(19,-207,["release","evidence"]),
 "ridge_rope_wear":(22,-207,["match","trial"]),
 "ridge_rope_records":(10,-211,["sequence","sources"]),
 "ridge_rope_credit":(20,-203,["credit","consent"]),
}
OUTCOMES={"life_lift_plan":("dual_safety","light_load"),"life_rope_future":("apprentice_pages","shared_lesson")}


def demonstrate(j):
    import copy
    from .journey import JourneyFailed
    if not j.walk_to(-.5,-143.5,y=109,radius=1.3,label="return to low lift model").ok:
        raise JourneyFailed("could not return to lift model")
    j.settle();baseline={};record_baseline(j,baseline)
    saved={q["id"]:copy.deepcopy(j.quest_entry(q["id"])) for q in j.snap().get("quests",[]) if q["id"].startswith(("life_lift_","life_rope_"))}
    for _ in range(2):
        j.drain_events();j.gw.call("landmark",landmarkId="ridge_lift_balance")
        node=j.wait_event("npc_dialog",lambda d:str(d.get("nodeId","")).startswith("life:ridge_lift_balance:demo:"))
        for option in ["loaded","tripped"]:
            j.gw.call("choose",entityId=node["entityId"],nodeId=node["nodeId"],optionId=option)
            node=j.wait_event("npc_dialog",lambda d,o=option:d.get("messageKey")=="life.ridge_lift_balance.demo.result."+o)
        j.gw.fire("close_dialog")
    rejoin(j);after={};record_baseline(j,after)
    if after!=baseline or any(j.quest_entry(q)!=entry for q,entry in saved.items()):
        raise JourneyFailed("lift demonstration changed saved quests or coins")
    return "raised and caught the load twice; close/reopen and reconnect; quests and coins unchanged"


def steps(j,roads,lift=0,rope=0,rope_first=False):
    from .journey import ROUTES
    north=[tuple(p) for p in roads["north"]]
    lower=[tuple(p) for p in roads["lower"]]
    upper=[tuple(p) for p in roads["upper"]]
    woods=[tuple(p) for p in roads["woods"]]
    baseline={}
    def go(area):
        if j.me()["z"] < -170 and area != "city":
            j.walk_route(list(reversed(upper))+[(-4,-148)],"down the upper switchbacks")
        if j.me()["z"] < -110 and area == "thorn":
            j.walk_route(list(reversed(lower))+list(reversed(north)),"down to Thornwatch")
        if j.me()["z"] > -110 and area != "thorn":
            j.walk_route(north+lower[1:]+[(-4,-148)],"up to Galepost")
        if j.me()["z"] > -170 and area == "city":
            j.walk_route(upper+[(8,-176)],"up to Windreach")
    def npc_area(n):
        return "thorn" if n=="sawyer_bettany" else "gale" if n=="lift_talric" else "city"
    def quest(q,n,action):
        go(npc_area(n));return j.quest(n,q,action)
    def practice(q,s):
        y=109 if s in ["ridge_lift_balance","ridge_lift_brake","ridge_lift_roles"] else 151
        go("gale" if y==109 else "city")
        return workshop(j,q,s,stations=STATIONS,feet_y=y,resume_station="ridge_lift_brake" if s.startswith("ridge_lift_") else "ridge_rope_records")
    outcomes={"life_lift_plan":OUTCOMES["life_lift_plan"][lift],"life_rope_future":OUTCOMES["life_rope_future"][rope]}
    def verify(n,q):
        go(npc_area(n));return verify_results(j,outcomes,baseline,stories=STORIES,reward=110,revisit_npc=n,revisit_quest=q)
    def replay():
        go("gale");return demonstrate(j)
    def return_city():
        go("thorn");return j.walk_route(list(reversed(woods))[:-1]+ROUTES["ironvein_to_city"]+[(128,128)],"return to Luminara by road")
    result=[("verify","record fresh character coins",lambda:record_baseline(j,baseline)),
            ("walk","causeway, woods road and complete mountain pass",lambda:j.walk_route([(128,116)]+ROUTES["city_to_ironvein"]+woods[1:]+[(-56,22)]+north+lower[1:]+[(-4,-148)],"Galepost on foot"))]
    ordered=STORIES[6:]+STORIES[:6] if rope_first else STORIES
    for q,giver,destination,station in ordered:
        result.append(("quest","accept "+q,lambda q=q,n=giver:quest(q,n,"accept")))
        if station:result.append(("workshop",station,lambda q=q,s=station:practice(q,s)))
        action="complete:"+outcomes[q] if q in outcomes else "complete"
        result.append(("quest","complete "+q,lambda q=q,n=destination,a=action:quest(q,n,a)))
    for n,q in [("ropewright_ossa","life_rope_future"),("lift_talric","life_lift_plan")]:
        result.append(("verify","saved results and revisit "+n,lambda n=n,q=q:verify(n,q)))
    result.append(("verify","repeat the finished lift model",replay))
    result.append(("walk","walk the return route to Luminara",return_city))
    return result
