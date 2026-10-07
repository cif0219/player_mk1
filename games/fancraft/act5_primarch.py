"""The gate guard: physical side approach, recovery console, real combat and a retained amendment."""
import time
from .city_life import rejoin, record_baseline
STORIES=[
 ('act5_primarch_briefing','seraine_kael','aldris_vane'),
 ('act5_primarch_sources','aldris_vane','recorder_tovan'),
 ('act5_primarch_retreat','recorder_tovan','roadwarden_ivara'),
 ('act5_primarch_limits','roadwarden_ivara','recorder_tovan'),
 ('act5_primarch','recorder_tovan','recorder_tovan'),
 ('act5_primarch_report','recorder_tovan','aldris_vane'),
]
def register_pass(j):
    from .journey import JourneyFailed
    j.drain_events();cursor=j.gw.event_cursor
    j.gw.call('enter',timeout_s=40,zone='story_primarch');time.sleep(1.5)
    start=time.monotonic()
    def state():return j.snap().get('primarch') or {}
    def walk(x,z):
        if not j.walk_to(x,z,y=67,radius=1.2,label='gate console approach').ok:raise JourneyFailed('cannot reach gate console')
    try:
        walk(70.5,65.5);j.stop()
        deadline=time.monotonic()+15
        while state().get('phase')!='warning' and time.monotonic()<deadline:time.sleep(.1)
        if state().get('phase')!='warning':raise JourneyFailed('guard never locked its direction')
        walk(70.5,54.5);walk(64.5,54.5);j.stop()
        deadline=time.monotonic()+15
        while state().get('phase')!='recovery' and time.monotonic()<deadline:time.sleep(.1)
        j.gw.call('landmark',landmarkId='primarch_console');time.sleep(.3)
        if not state().get('archived'):raise JourneyFailed('old order not copied during recovery')
        target=j.entity(lambda e:e.get('mobId')=='sentinel_primarch' and e.get('alive'))
        if not target or not j.engage(target,timeout_s=150):raise JourneyFailed('guard combat failed')
        time.sleep(.2)
        if state().get('phase')!='halted':raise JourneyFailed('guard death did not retain a halted console')
        if any(e.type=='dungeon_complete' for e in j.gw.events_since(cursor)):raise JourneyFailed('death alone incorrectly granted passage')
        walk(64.5,54.5);j.stop();j.gw.call('landmark',landmarkId='primarch_console');time.sleep(.4)
        if state().get('phase')!='cleared':raise JourneyFailed('checked pass not registered')
        evs=j.gw.events_since(cursor)
        if sum(e.type=='dungeon_complete' for e in evs)!=1:raise JourneyFailed('completion missing or duplicated')
        if any(e.type=='instance_reset' for e in evs):raise JourneyFailed('guard encounter reset')
        j.report.kills['sentinel_primarch']=j.report.kills.get('sentinel_primarch',0)+1
        return f'copied during recovery; real guard kill; death alone did not clear; registered pass once; {time.monotonic()-start:.1f}s'
    finally:
        j.stop();j.gw.call('enter',timeout_s=40,zone='overworld');time.sleep(2)

def steps(j,roads,choice='short_pulses'):
    from .journey import FROM_ASSEMBLY, TO_ASSEMBLY, WINDREACH_CRYSTAL, JourneyFailed
    baseline={};prior={};paths={k:[tuple(p) for p in v] for k,v in roads.items()}
    outer=sum([paths[k] for k in ('coast','aid','edge','scarPass','scarSurvey','pillars')],[])
    outer_back=sum([list(reversed(paths[k])) for k in ('pillars','scarSurvey','scarPass','edge','aid','coast')],[])
    inner=paths['ring']+paths['ante'];inner_back=list(reversed(paths['ante']))+list(reversed(paths['ring']))
    ids={q for q,*_ in STORIES}
    def old_rows():return {q['id']:(q.get('state'),q.get('have'),q.get('choiceId')) for q in j.snap().get('quests',[]) if q['id'] not in ids and q['id'].startswith(('act4_','act5_','life_')) and q['state']!='available'}
    def check_start():
        if j.quest_state('act5_swarm_report')!='completed':raise JourneyFailed('requires the saved Swarm report')
        plan=j.quest_entry('act4_scar_plan')
        if plan and plan.get('choiceId') is not None and plan['choiceId']!=choice:raise JourneyFailed('prior plan changed')
        prior.update(old_rows());return record_baseline(j,baseline)
    def kit():
        if j.me()['z']<95 and j.me()['x']>100 and j.me()['x']<150:
            if not j.walk_route(FROM_ASSEMBLY,'leave the Assembly'):raise JourneyFailed('assembly exit failed')
        if j.quest_state('guild_armament')!='completed':
            if j.quest_state('guild_orientation')!='completed':
                if j.quest_state('guild_orientation') not in ('active','ready'):j.quest('guild_mira','guild_orientation','accept',via=[(152,116),(182,116),(201,112),(201,102),(194,94)])
                j.quest('recruit_ren','guild_orientation','complete')
            if j.quest_state('guild_armament') not in ('active','ready'):j.quest('recruit_ren','guild_armament','accept')
            j.quest('guild_mira','guild_armament','complete')
        return j.equip_items([('guild_sword','left'),('guild_shield','right')])
    def walk(route,label):
        if not j.walk_route(route,label):raise JourneyFailed(label+' failed')
        return label
    def home():
        if j.me()['z']<-175:
            walk([(8,-203),(8,-196),(8,-190)],'leave the Sanctum');j.travel(WINDREACH_CRYSTAL,'luminara')
        if j.me()['z']<95 and 100<j.me()['x']<150:walk(FROM_ASSEMBLY,'leave the Assembly')
        return 'returned to Luminara for the next shift'
    def verify():
        rejoin(j)
        if any(j.quest_state(q)!='completed' for q,*_ in STORIES):raise JourneyFailed('completion lost')
        if old_rows()!=prior:raise JourneyFailed('previous records changed')
        if j.me()['coins']-baseline['coins']!=900:raise JourneyFailed('incorrect rewards')
        return 'six saved quests, 900 coins once, all prior records unchanged'
    result=[('verify','continue saved Swarm report',check_start),('travel','return from last visit if needed',home),('equipment','ordinary guild sword and shield',kit)]
    for q,giver,dest in STORIES:
        result.append(('quest','accept '+q,lambda q=q,n=giver:j.quest(n,q,'accept',via=[(182,116),(201,115)] if q=='act5_primarch_briefing' else None)))
        if q=='act5_primarch_briefing':result.append(('walk','take the command to Vane',lambda:walk([(201,115),(182,115),(128,116)]+TO_ASSEMBLY,'Assembly Hall')))
        if q=='act5_primarch_sources':result.append(('walk','take the signed draft to Tovan',lambda:walk(FROM_ASSEMBLY+[(182,116),(182,149)]+outer+inner,'southern road to Tovan')))
        if q=='act5_primarch_retreat':result.append(('walk','confirm the retreat limits',lambda:walk(inner_back,'return to Ivara')))
        if q=='act5_primarch_limits':result.append(('walk','bring the retreat page to Tovan',lambda:walk(inner,'inner approach')))
        if q=='act5_primarch':result.append(('combat','copy old command, disable guard and register pass',lambda:register_pass(j)))
        if q=='act5_primarch_report':result.append(('walk','bring the receipt home',lambda:walk(inner_back+outer_back+[(182,149),(182,116),(128,116)]+TO_ASSEMBLY,'return to Vane')))
        result.append(('quest','complete '+q,lambda q=q,n=dest:j.quest(n,q,'complete')))
    result.append(('verify','rejoin and check saved records',verify))
    return result
