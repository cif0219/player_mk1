"""Resume early main story through ordinary gameplay; no quest-skip commands or save writes."""
import time
from .city_life import rejoin

EARLY = """petra_welcome story_manifest story_beacon story_archive story_report
kael_training kael_hollow act1_chitin act1_crystal act1_cull act1_heart
act2_descent act2_signal act2_gallery act2_words
act3_charter act3_woods_road act3_stone_reading act3_north_road act3_foothills
act3_signal act3_stormbreak act3_galepost_signal act3_windreach act3_two_ends
act3_open_method act3_missing_pages act3_pinnacle_crystal act3_tempest
act3_ship_log act3_cross_check act3_cut_pages act3_convene
act4_departure act4_supply act4_triage act4_return_marks act4_embers""".split()

def steps(j, roads, finale=True):
    from .journey import (JourneyFailed, KEEP, prologue_steps, guild_steps, act1_steps,
                          act2_steps, act3_steps, accord_steps, ACCORD_ECHOES)
    def refresh():
        j.gw.fire("snapshot");time.sleep(.35)
    refresh()
    prior={q["id"]:q.get("choiceId") for q in j.snap().get("quests",[]) if q["state"]=="completed"}
    original_quest=j.quest
    def quest(npc, qid, action, **kw):
        refresh();state=j.quest_state(qid)
        if state=="completed" or (action=="accept" and state in ("active","ready")):
            return "already saved: "+qid+" "+state
        result=original_quest(npc,qid,action,**kw);refresh();return result
    j.quest=quest
    zones={"verdant_hollow":"kael_hollow", "story_hollow_heart":"act1_heart", "story_tempest":"act3_tempest",
           **{zone:qid for qid,zone,_ in ACCORD_ECHOES}}
    original_clear=j.clear_instance
    def clear(zone,boss,**kw):
        refresh()
        if j.quest_state(zones.get(zone,"")) in ("ready","completed"):return "clear already saved"
        return original_clear(zone,boss,**kw)
    j.clear_instance=clear
    j.instance=lambda zone,boss,**kw:bool(clear(zone,boss,**kw))
    original_mine=j.mine
    j.mine=lambda *a,**kw:"samples already delivered" if j.quest_state("act2_signal")=="completed" else original_mine(*a,**kw)
    original_collect=j.collect_drops
    j.collect_drops=lambda *a,**kw:"chitin already delivered" if a[0]=="hollow_chitin" and j.quest_state("act1_chitin")=="completed" else original_collect(*a,**kw)
    def walk(route,label):
        if not j.walk_route(route,label):raise JourneyFailed(label)
        return label
    result=[]
    if j.quest_state("story_report")!="completed":
        result.append(("walk","Keep road to the arrival lawn",lambda:walk([(201,116),(182,116),(152,116),(128,116),(128,128)],"arrival lawn")))
        result+=prologue_steps(j)
    if j.quest_state("guild_armament")!="completed":result+=guild_steps(j)
    else:result.append(("equipment","ordinary saved guild weapons",lambda:j.equip_items([("guild_sword","left"),("guild_shield","right")])))
    if j.quest_state("act1_heart")!="completed":result+=act1_steps(j,roads)
    if j.quest_state("act2_words") in ("active","ready"):
        from .journey import DEEP_LIGHT_CRYSTAL
        result += [("travel","resume Deep Light return",lambda:j.travel(DEEP_LIGHT_CRYSTAL,"luminara")),
                   ("quest","Tethis receives the saved report",lambda:quest("tethis","act2_words","complete"))]
    elif j.quest_state("act2_words")!="completed":result+=act2_steps(j,roads)
    if j.quest_state("act3_convene")!="completed":result+=act3_steps(j,roads,gear=False)
    coast=[tuple(p) for p in roads["coast"]];aid=[tuple(p) for p in roads["aid"]];edge=[tuple(p) for p in roads["edge"]]
    if j.quest_state("act4_embers")!="completed":
        result += [
          ("quest","Kael: the south road",lambda:quest("seraine_kael","act4_departure","accept")),
          ("quest","Vesa: supplies",lambda:quest("quartermaster_vesa","act4_departure","complete",via=[(201,115),(182,115),(182,149)]+coast)),
          ("quest","accept supply ledger",lambda:quest("quartermaster_vesa","act4_supply","accept")),
          ("landmark","read supply ledger",lambda:j.landmark("southern_supply_ledger",245,229)),
          ("quest","deliver supply ledger",lambda:quest("quartermaster_vesa","act4_supply","complete")),
          ("quest","accept aid station",lambda:quest("quartermaster_vesa","act4_triage","accept")),
          ("quest","Maelin: beds and return route",lambda:quest("medic_maelin","act4_triage","complete",via=aid)),
          ("quest","accept return board",lambda:quest("medic_maelin","act4_return_marks","accept")),
          ("landmark","read return board",lambda:j.landmark("southern_return_board",267,269)),
          ("quest","confirm return route",lambda:quest("medic_maelin","act4_return_marks","complete")),
          ("quest","accept Ember's Edge",lambda:quest("medic_maelin","act4_embers","accept")),
          ("quest","Corrin receives both records",lambda:quest("marshal_corrin","act4_embers","complete",via=edge)),
          ("quest","accept southern report",lambda:quest("marshal_corrin","act4_report","accept")),
          ("walk","walk the south road home",lambda:walk(list(reversed(edge))+list(reversed(aid))+list(reversed(coast))+[(182,149),(182,115),(201,115)],"south road home")),
          ("quest","deliver southern report",lambda:quest("seraine_kael","act4_report","complete")),
        ]
    if finale and j.quest_state("accord_home")!="completed":
        result.append(("walk","Keep gate for Seven Echoes",lambda:walk(KEEP,"Keep gate")))
        result+=accord_steps(j)
    def verify_early():
        rejoin(j);missing=[qid for qid in EARLY if j.quest_state(qid)!="completed"]
        if missing:raise JourneyFailed("unfinished early story: "+str(missing))
        now={q["id"]:q.get("choiceId") for q in j.snap().get("quests",[]) if q["state"]=="completed"}
        if any(now.get(k)!=v or k not in now for k,v in prior.items()):raise JourneyFailed("prior completed records changed")
        return "38 early quests saved; earlier choices and completed chapters preserved"
    result.append(("verify","early story durable after reconnect",verify_early))
    if finale:
        result.append(("quest","deliver the final three-city report",lambda:quest("seraine_kael","act5_accord_report","complete")))
        def verify_final():
            rejoin(j)
            ids={q["id"] for q in j.snap().get("quests",[]) if q.get("chapter") in ("prologue","act1","act2","act3","act4","act5","accord") and q["state"]=="completed"}
            if len(ids)!=112 or j.quest_state("act5_accord_report")!="completed":raise JourneyFailed("expected 112 saved public quests; found "+str(len(ids)))
            return "112 public quests complete and durable; no test commands used"
        result.append(("verify","complete public campaign after reconnect",verify_final))
    return result
