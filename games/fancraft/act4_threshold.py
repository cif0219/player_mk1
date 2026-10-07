"""Continue a saved Meltscar character on foot. No setup or privileged commands."""
from .city_life import record_baseline, rejoin

STORIES = [
 ("act4_pillars_orders","seraine_kael","surveyor_sevin",None),
 ("act4_pillars_route","surveyor_sevin","roadwarden_ivara","pillar_route"),
 ("act4_pillars_cover","roadwarden_ivara","roadwarden_ivara","pillar_cover"),
 ("act4_pillars_record","roadwarden_ivara","roadwarden_ivara","pillar_records"),
 ("act4_pillars_last","roadwarden_ivara","roadwarden_ivara","pillar_tail"),
 ("act4_threshold_entry","roadwarden_ivara","recorder_tovan",None),
 ("act4_threshold_bindings","recorder_tovan","recorder_tovan","threshold_bindings"),
 ("act4_threshold_method","recorder_tovan","recorder_tovan","threshold_method"),
 ("act4_threshold_return","recorder_tovan","medic_maelin",None),
 ("act4_threshold_report","medic_maelin","seraine_kael",None),
]
STATIONS={"pillar_route":(381,407,87,["arrows","turn"]),"pillar_cover":(391,414,87,["screen","gap","reply"]),
 "pillar_records":(419,435,91,["sequence","source"]),"pillar_tail":(382,414,87,["roster","arrival","clear"]),
 "threshold_bindings":(448,445,95,["core","binding","unknown"]),"threshold_method":(454,438,95,["restart","divert","rebind"])}

def practice(j,q,station,choice):
    from .journey import JourneyFailed
    x,z,y,_=STATIONS[station]
    if not j.walk_to(x+.5,z+1.5,y=y,radius=.45,label=station).ok:raise JourneyFailed("cannot reach "+station)
    j.settle()
    pos=j.me().get("server") or j.me()
    if abs(pos["y"]-y)>2:raise JourneyFailed(f"wrong station floor: {station} {pos}")
    def open_node():
        j.drain_events();j.gw.call("landmark",landmarkId=station)
        return j.wait_event("npc_dialog",lambda d:str(d.get("nodeId","")).startswith("life:"+station+":"))
    def choose(node,option):
        j.gw.call("choose",entityId=node["entityId"],nodeId=node["nodeId"],optionId=option)
        return j.wait_event("npc_dialog",lambda d:str(d.get("nodeId","")).startswith("life:"+station+":"))
    node=open_node();steps=STATIONS[station][3]
    for i,answer in enumerate(steps):
        if node["nodeId"].split(":")[2]!=str(i):raise JourneyFailed("wrong saved step")
        node=choose(node,"reconsider")
        if node["nodeId"].split(":")[2]!=str(i) or ".feedback." not in node["messageKey"]:raise JourneyFailed("wrong answer advanced the test")
        node=choose(node,answer)
        if i==0 and station in ("pillar_records","pillar_tail","threshold_method"):
            j.gw.fire("close_dialog");rejoin(j)
            if j.quest_entry(q).get("have")!=1:raise JourneyFailed("partial record lost")
            node=open_node()
    j.gw.fire("close_dialog")
    j.gw.wait_snapshot(lambda d:any(row["id"]==q and row["state"]=="ready" for row in d.get("quests",[])),timeout_s=8)
    return str(len(steps))+" observations; retry checked; progress saved"

def steps(j,roads,choice="short_pulses"):
    from .journey import KEEP, JourneyFailed
    baseline={};prior={}
    paths={name:[tuple(p) for p in roads[name]] for name in ("coast","aid","edge","scarPass","scarSurvey","pillars","ring","ante")}
    def old_rows():
        return {q["id"]:(q.get("state"),q.get("have"),q.get("choiceId")) for q in j.snap().get("quests",[]) if q["id"].startswith("act4_scar_")}
    def check_start():
        if j.quest_state("act4_scar_report")!="completed":raise JourneyFailed("requires a completed Meltscar report; scenario never skips")
        plan=j.quest_entry("act4_scar_plan")
        if plan and plan.get("choiceId") is not None and plan.get("choiceId")!=choice:raise JourneyFailed("unexpected previous plan")
        if any(q.get("state") in ("active","ready","completed") and q.get("chapter")=="side" for q in j.snap().get("quests",[])):raise JourneyFailed("requires no side-story progress")
        prior.update(old_rows())
        return record_baseline(j,baseline)
    def verify():
        rejoin(j)
        if any(j.quest_state(q)!="completed" for q,*_ in STORIES):raise JourneyFailed("completed quest lost")
        if old_rows()!=prior:raise JourneyFailed("previous Meltscar progress changed")
        if j.me()["coins"]-baseline["coins"]!=1100:raise JourneyFailed("wrong total reward")
        if any(q.get("state") in ("active","ready","completed") and q.get("chapter")=="side" for q in j.snap().get("quests",[])):raise JourneyFailed("main story touched a side story")
        return "10 saved quests; 1100 coins exactly once; prior Meltscar records unchanged; no side stories; returned on foot"
    result=[("verify","continue saved chapter without side stories",check_start)]
    for q,giver,dest,station in STORIES:
        via=KEEP if q=="act4_pillars_orders" else None
        result.append(("quest","accept "+q,lambda q=q,n=giver,v=via:j.quest(n,q,"accept",via=v)))
        route=paths["pillars"] if q=="act4_pillars_route" else paths["ring"] if q=="act4_pillars_record" else None
        if route:result.append(("walk","approach "+str(station),lambda r=route:j.walk_route(r,"black-pillar approach")))
        if station:result.append(("survey",station,lambda q=q,s=station:practice(j,q,s,choice)))
        via=None
        if q=="act4_pillars_orders":via=[(201,115),(182,115),(182,149)]+sum([paths[k] for k in ("coast","aid","edge","scarPass","scarSurvey")],[])
        if q=="act4_pillars_record":via=list(reversed(paths["ring"]))
        if q=="act4_threshold_entry":via=paths["ring"]+paths["ante"]
        if q=="act4_threshold_return":via=sum([list(reversed(paths[k])) for k in ("ante","ring","pillars","scarSurvey","scarPass","edge")],[])
        if q=="act4_threshold_report":via=list(reversed(paths["aid"]))+list(reversed(paths["coast"]))+[(182,149),(182,115),(201,115)]
        result.append(("quest","complete "+q,lambda q=q,n=dest,v=via:j.quest(n,q,"complete",via=v)))
    result.append(("verify","saved records and rewards after return",verify))
    return result
