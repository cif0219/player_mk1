"""S22 acceptance on real paths and ordinary dialogue, before or after the campaign."""
import copy
from .city_life import record_baseline, workshop, rejoin
STORIES=[
 ("life_wagon_space","quartermaster_vesa","quartermaster_vesa","supply_wagon_space"),
 ("life_wagon_wind","quartermaster_vesa","quartermaster_vesa","supply_wagon_wind"),
 ("life_wagon_drill","quartermaster_vesa","medic_maelin","supply_wagon_drill"),
 ("life_wagon_clips","medic_maelin","lift_talric",None),
 ("life_wagon_diagram","lift_talric","quartermaster_vesa","supply_wagon_diagram"),
 ("life_wagon_plan","quartermaster_vesa","quartermaster_vesa",None),
]
STATIONS={"supply_wagon_space":(256,215,["wheels","stretcher","access"]),
 "supply_wagon_wind":(261,215,["broad","small"]),
 "supply_wagon_drill":(256,221,["broad_fold","broad_clip","broad_pass","small_fold","small_clip","small_pass"]),
 "supply_wagon_diagram":(261,221,["marks","handover"])}

def steps(j,roads,layout="broad_shade"):
 from .journey import JourneyFailed, ROUTES
 coast=[tuple(p) for p in roads["coast"]];aid=[tuple(p) for p in roads["aid"]]
 woods=[tuple(p) for p in roads["woods"]];north=[tuple(p) for p in roads["north"]];lower=[tuple(p) for p in roads["lower"]]
 baseline={};old={};outcomes={"life_wagon_plan":layout}
 def start():
  record_baseline(j,baseline)
  old.update({q["id"]:copy.deepcopy(q) for q in j.snap().get("quests",[]) if q.get("state") in ("active","ready","completed") and not q["id"].startswith("life_wagon_")})
  if any(j.quest_state(q) in ("active","ready","completed") for q,*_ in STORIES):raise JourneyFailed("S22 scenario requires a character who has not started S22")
  return f"preserve {len(old)} prior records and {baseline['coins']} coins"
 def road(points,label):
  if not j.walk_route(points,label):raise JourneyFailed("route failed: "+label)
 def go(area):
  if j.me()["z"] < -110:
   if area=="gale":return
   road(list(reversed(lower))+list(reversed(north))+[(-56,22),(-44,30)]+list(reversed(woods))[:-1]+ROUTES["ironvein_to_city"]+[(128,116)],"return from Galepost by road")
  # Leave the annex through the level joint to the existing depot pad.
  if j.me()["x"]>=255 and 213<=j.me()["z"]<=224:
   road([(255.5,222.5),(254,222),(254,224),(250,224)],"leave wagon yard")
  if j.me()["z"]>240 and area!="aid":road(list(reversed(aid)),"return from aid station")
  if area in ("city","gale") and j.me()["z"]>210:
   road([(248,226)]+list(reversed(coast))+[(182,149),(182,115),(128,116)],"coast road back to city")
  if area=="gale":
   road([(128,116)]+ROUTES["city_to_ironvein"]+woods[1:]+[(-56,22)]+north+lower[1:]+[(-4,-148)],"woods and mountain road to Talric")
  if area in ("supply","aid") and j.me()["z"]<210:
   road([(150,115),(182,115),(182,149)]+coast,"coast road to Vesa")
  if area=="aid" and j.me()["z"]<240:road(aid,"road to Maelin")
 def area(n):return "gale" if n=="lift_talric" else "aid" if n=="medic_maelin" else "supply"
 def quest(q,n,action):
  go(area(n));return j.quest(n,q,action)
 def practice(q,s):
  go("supply")
  npc=j.approach_npc("quartermaster_vesa")
  y=npc.get("y",npc.get("position",{}).get("y",j.me()["y"]))
  road([(254,224),(254,222),(255.5,222.5)],"enter wagon practice yard")
  return workshop(j,q,s,stations=STATIONS,feet_y=y,resume_station="supply_wagon_drill")
 def verify():
  rejoin(j);record_baseline(j,{})
  for q,*_ in STORIES:
   if j.quest_state(q)!="completed":raise JourneyFailed(q+" lost after reconnect")
  for q,c in outcomes.items():
   if j.quest_entry(q).get("choiceId")!=c:raise JourneyFailed("lost choice "+q)
  for q,entry in old.items():
   if j.quest_entry(q)!=entry:raise JourneyFailed("changed prior quest "+q)
  if j.me()["coins"]-baseline["coins"]!=60:raise JourneyFailed("S22 rewards must total exactly 60 coins")
  return f"six saved quests, choice {layout}, 60 coins once; {len(old)} old records unchanged"
 def revisit(n):
  go(area(n))
  npc=j.approach_npc(n);expected=f"quest.life_wagon_plan.revisit.{layout}.{n}"
  for _ in range(14):
   j.drain_events();j.gw.call("talk",entityId=npc["entityId"]);msg=j.wait_event("npc_dialog")
   if msg.get("nodeId")=="quest_menu":
    if any(o["id"]=="life_wagon_plan" for o in msg.get("options",[])):raise JourneyFailed("repeat reward offered")
    j.gw.call("choose",entityId=npc["entityId"],nodeId=msg["nodeId"],optionId="talk");msg=j.wait_event("npc_dialog")
   j.gw.fire("close_dialog")
   if msg.get("messageKey")==expected:return expected
  raise JourneyFailed("missing S22 revisit at "+n)
 def demonstrate():
  go("supply")
  road([(254,224),(254,222),(255.5,222.5)],"return to wagon models")
  for station,options in [("supply_wagon_wind",["broad","small"]),("supply_wagon_drill",["folded","ready"])]:
   x,z,_=STATIONS[station]
   if not j.walk_to(x+.5,z+.5,y=j.me()["y"],radius=1.3,label=station).ok:raise JourneyFailed("cannot reach demonstration")
   j.settle();j.drain_events();j.gw.call("landmark",landmarkId=station)
   node=j.wait_event("npc_dialog",lambda d:str(d.get("nodeId","")).startswith("life:"+station+":demo:"))
   for option in options*2:
    j.gw.call("choose",entityId=node["entityId"],nodeId=node["nodeId"],optionId=option)
    node=j.wait_event("npc_dialog",lambda d:d.get("messageKey")==f"life.{station}.demo.result.{option}")
   j.gw.fire("close_dialog")
  return verify()
 result=[("verify","capture all previous story choices",start)]
 for q,giver,dest,station in STORIES:
  result.append(("quest","accept "+q,lambda q=q,n=giver:quest(q,n,"accept")))
  if station:result.append(("workshop",station,lambda q=q,s=station:practice(q,s)))
  action="complete:"+outcomes[q] if q in outcomes else "complete"
  result.append(("quest","complete "+q,lambda q=q,n=dest,a=action:quest(q,n,a)))
 result.append(("verify","saved S22 and original records",verify))
 result.append(("verify","repeat demonstrations without rewards",demonstrate))
 for n in ["quartermaster_vesa","medic_maelin","lift_talric"]:result.append(("verify","personal revisit "+n,lambda n=n:revisit(n)))
 result.append(("walk","return to the city by road",lambda:go("city")))
 result.append(("verify","returned to city with all choices",verify))
 return result
