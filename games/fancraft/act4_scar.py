"""Act IV acceptance. Requires an explicitly prepared act4_report save; never skips inside the scenario."""
from .city_life import record_baseline, rejoin

STORIES = [
 ("act4_scar_orders","seraine_kael","marshal_corrin",None),
 ("act4_scar_route","marshal_corrin","surveyor_sevin","scar_wayback"),
 ("act4_scar_baseline","surveyor_sevin","surveyor_sevin","scar_baseline"),
 ("act4_scar_plan","surveyor_sevin","surveyor_sevin",None),
 ("act4_scar_clear","surveyor_sevin","surveyor_sevin","scar_clearance"),
 ("act4_scar_isolate","surveyor_sevin","surveyor_sevin","scar_isolator"),
 ("act4_scar_recovery","surveyor_sevin","surveyor_sevin","scar_recovery"),
 ("act4_scar_limit","surveyor_sevin","medic_maelin",None),
 ("act4_scar_report","medic_maelin","seraine_kael",None),
]
STATIONS={"scar_wayback":(318,349,75,["route","shelter"]),"scar_baseline":(354,372,83,["labels","baseline"]),
 "scar_clearance":(344,373,83,[]),"scar_isolator":(356,379,83,[]),"scar_recovery":(346,382,83,["both","unknown"])}
def answers(station,choice):
    if station=="scar_clearance":return ["roster","shelter","signal"] if choice=="short_pulses" else ["roster","shelter","second_count","signal"]
    if station=="scar_isolator":return ["limit","pulse_one","cool","pulse_two","restore"] if choice=="short_pulses" else ["window","isolate","stop","restore"]
    return STATIONS[station][3]

def practice(j,q,station,choice):
    from .journey import JourneyFailed
    x,z,y,_=STATIONS[station]
    if not j.walk_to(x+.5,z+.5,y=y,radius=1.35,label=station).ok:raise JourneyFailed("cannot reach "+station)
    j.settle()
    def open_node():
        j.drain_events();j.gw.call("landmark",landmarkId=station)
        return j.wait_event("npc_dialog",lambda d:str(d.get("nodeId","")).startswith("life:"+station+":"))
    def choose(node,option):
        j.gw.call("choose",entityId=node["entityId"],nodeId=node["nodeId"],optionId=option)
        return j.wait_event("npc_dialog",lambda d:str(d.get("nodeId","")).startswith("life:"+station+":"))
    node=open_node();steps=answers(station,choice)
    for i,answer in enumerate(steps):
        if node["nodeId"].split(":")[2]!=str(i):raise JourneyFailed("wrong saved step")
        node=choose(node,"reconsider")
        if node["nodeId"].split(":")[2]!=str(i) or ".feedback." not in node["messageKey"]:raise JourneyFailed("wrong answer advanced the test")
        node=choose(node,answer)
        if i==0 and station in ("scar_clearance","scar_isolator"):
            j.gw.fire("close_dialog");rejoin(j)
            if j.quest_entry(q).get("have")!=1:raise JourneyFailed("partial record lost")
            node=open_node()
    if station=="scar_isolator":
        node=choose(node,"restart_trial")
        if node["nodeId"].split(":")[2]!="0":raise JourneyFailed("restart failed")
        for answer in steps:node=choose(node,answer)
    j.gw.fire("close_dialog")
    j.gw.wait_snapshot(lambda d:any(row["id"]==q and row["state"]=="ready" for row in d.get("quests",[])),timeout_s=8)
    return str(len(steps))+" observations; retry checked"+("; saved midway, then repeated before submission" if station=="scar_isolator" else "")

def steps(j,roads,choice="short_pulses"):
    from .journey import KEEP, JourneyFailed
    baseline={}
    coast=[tuple(p) for p in roads["coast"]];aid=[tuple(p) for p in roads["aid"]];edge=[tuple(p) for p in roads["edge"]]
    passage=[tuple(p) for p in roads["scarPass"]];survey=[tuple(p) for p in roads["scarSurvey"]]
    def check_start():
        if j.quest_state("act4_report")!="completed":raise JourneyFailed("prepare an act4_report test save first; scenario never skips chapters")
        if any(q.get("state") in ("active","ready","completed") and q.get("chapter")=="side" for q in j.snap().get("quests",[])):raise JourneyFailed("test requires no side-story progress")
        return record_baseline(j,baseline)
    def verify():
        rejoin(j)
        if any(j.quest_state(q)!="completed" for q,*_ in STORIES):raise JourneyFailed("completed quest lost")
        if j.quest_entry("act4_scar_plan").get("choiceId")!=choice:raise JourneyFailed("plan lost")
        if j.me()["coins"]-baseline["coins"]!=920:raise JourneyFailed("wrong total reward")
        if any(q.get("state") in ("active","ready","completed") and q.get("chapter")=="side" for q in j.snap().get("quests",[])):raise JourneyFailed("main story touched a side story")
        return "9 saved quests; plan "+choice+"; exactly 920 coins; no side-story progress; returned on foot"
    result=[("verify","prepared chapter save without side stories",check_start)]
    for q,giver,dest,station in STORIES:
        via=KEEP if q=="act4_scar_orders" else None
        result.append(("quest","accept "+q,lambda q=q,n=giver,v=via:j.quest(n,q,"accept",via=v)))
        if q=="act4_scar_route":result.append(("walk","southern gate to return shelter",lambda:j.walk_route(passage,"Meltscar pass")))
        if station:result.append(("survey",station,lambda q=q,s=station:practice(j,q,s,choice)))
        via=None
        if q=="act4_scar_orders":via=[(201,115),(182,115),(182,149)]+coast+aid+edge
        if q=="act4_scar_route":via=survey
        if q=="act4_scar_limit":via=list(reversed(survey))+list(reversed(passage))+list(reversed(edge))
        if q=="act4_scar_report":via=list(reversed(aid))+list(reversed(coast))+[(182,149),(182,115),(201,115)]
        action="complete:"+choice if q=="act4_scar_plan" else "complete"
        result.append(("quest","complete "+q,lambda q=q,n=dest,a=action,v=via:j.quest(n,q,a,via=v)))
    result.append(("verify","durable choices and rewards after round trip",verify))
    return result
