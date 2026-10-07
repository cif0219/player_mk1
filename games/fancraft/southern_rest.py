"""S23 acceptance on real paths and ordinary dialogue, before or after the campaign."""
import copy
from .city_life import record_baseline, workshop, rejoin
STORIES=[
 ("life_rest_cards","medic_maelin","medic_maelin","aid_rest_cards"),
 ("life_rest_handover","medic_maelin","healer_iona",None),
 ("life_rest_contact","healer_iona","marshal_corrin","aid_rest_contact"),
 ("life_rest_lane","marshal_corrin","medic_maelin","aid_rest_lane"),
 ("life_rest_pause","medic_maelin","medic_maelin","aid_rest_pause"),
 ("life_rest_corner","medic_maelin","medic_maelin",None),
]
STATIONS={"aid_rest_cards":(270,261,["directions","rest","medical"]),"aid_rest_contact":(275,269,["roles","repeat"]),
 "aid_rest_lane":(270,270,["entrance","rear"]),"aid_rest_pause":(274,272,["shelf","ask"])}

def steps(j,roads,layout="duty_seat",pause="warm_cup"):
 from .journey import JourneyFailed
 coast=[tuple(p) for p in roads["coast"]];aid=[tuple(p) for p in roads["aid"]];edge=[tuple(p) for p in roads["edge"]]
 baseline={};old={};outcomes={"life_rest_corner":layout,"life_rest_pause":pause}
 def start():
  record_baseline(j,baseline)
  old.update({q["id"]:copy.deepcopy(q) for q in j.snap().get("quests",[]) if q.get("state") in ("active","ready","completed") and not q["id"].startswith("life_rest_")})
  if any(j.quest_state(q) in ("active","ready","completed") for q,*_ in STORIES):raise JourneyFailed("S23 scenario requires a character who has not started S23")
  return f"preserve {len(old)} prior records and {baseline['coins']} coins"
 def road(points,label):
  if not j.walk_route(points,label):raise JourneyFailed("route failed: "+label)
 def go(area):
  if j.me()["z"]>290:road(list(reversed(edge)),"return from Ember's Edge")
  if area=="city" and j.me()["z"]>220:road(list(reversed(aid))+list(reversed(coast))+[(182,149),(182,115),(150,115),(150,122)],"coast road back to Iona")
  if area!="city" and j.me()["z"]<220:road([(150,115),(182,115),(182,149)]+coast+aid,"walk to the white aid awning")
  if area=="edge":road(edge,"road to Corrin")
 def quest(q,n,action):
  go("city" if n=="healer_iona" else "edge" if n=="marshal_corrin" else "aid")
  return j.quest(n,q,action)
 def practice(q,s):
  go("aid")
  # The aid pad follows generated terrain; the visible medic is authoritative for its floor.
  npc=j.approach_npc("medic_maelin")
  y=npc.get("y",npc.get("position",{}).get("y",j.me()["y"]))
  return workshop(j,q,s,stations=STATIONS,feet_y=y,resume_station="aid_rest_contact")
 def verify():
  rejoin(j);record_baseline(j,{})
  for q,*_ in STORIES:
   if j.quest_state(q)!="completed":raise JourneyFailed(q+" lost after reconnect")
  for q,c in outcomes.items():
   if j.quest_entry(q).get("choiceId")!=c:raise JourneyFailed("lost choice "+q)
  for q,entry in old.items():
   if j.quest_entry(q)!=entry:raise JourneyFailed("changed prior quest "+q)
  if j.me()["coins"]-baseline["coins"]!=60:raise JourneyFailed("S23 rewards must total exactly 60 coins")
  return f"six saved quests, choices {layout}/{pause}, 60 coins once; {len(old)} old records unchanged"
 def revisit(n):
  go("city" if n=="healer_iona" else "edge" if n=="marshal_corrin" else "aid")
  npc=j.approach_npc(n);expected=f"quest.life_rest_corner.revisit.{layout}.{n}"
  for _ in range(14):
   j.drain_events();j.gw.call("talk",entityId=npc["entityId"]);msg=j.wait_event("npc_dialog")
   if msg.get("nodeId")=="quest_menu":
    if any(o["id"]=="life_rest_corner" for o in msg.get("options",[])):raise JourneyFailed("repeat reward offered")
    j.gw.call("choose",entityId=npc["entityId"],nodeId=msg["nodeId"],optionId="talk");msg=j.wait_event("npc_dialog")
   j.gw.fire("close_dialog")
   if msg.get("messageKey")==expected:return expected
  raise JourneyFailed("missing S23 revisit at "+n)
 def demonstrate():
  go("aid")
  for station,options in [("aid_rest_lane",["entrance","rear"]),("aid_rest_pause",["cup","game"])]:
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
 result.append(("verify","saved S23 and original records",verify))
 result.append(("verify","repeat demonstrations without rewards",demonstrate))
 for n in ["medic_maelin","marshal_corrin","healer_iona"]:result.append(("verify","personal revisit "+n,lambda n=n:revisit(n)))
 result.append(("verify","returned to city with all choices",verify))
 return result
