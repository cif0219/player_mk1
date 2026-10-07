"""The inner windows: ordinary movement, eight real adds, work circles and hive combat."""
import time
import math
from .city_life import rejoin, record_baseline

STORIES=[
 ('act5_swarm_briefing','seraine_kael','roadwarden_ivara'),
 ('act5_swarm_route','roadwarden_ivara','recorder_tovan'),
 ('act5_swarm','recorder_tovan','recorder_tovan'),
 ('act5_swarm_readings','recorder_tovan','roadwarden_ivara'),
 ('act5_swarm_report','roadwarden_ivara','seraine_kael'),
]

def protect_windows(j,timeout_s=360):
    from .journey import JourneyFailed
    j.drain_events();cursor=j.gw.event_cursor
    j.gw.call('enter',timeout_s=40,zone='story_hollow_swarm');time.sleep(1.5)
    start=time.monotonic();phases=[];kills=0;cleared=False;resets=0
    try:
        while time.monotonic()-start<timeout_s:
            evs=j.gw.events_since(cursor)
            if evs:cursor=evs[-1].seq
            if any(e.type=='instance_reset' for e in evs):
                resets+=1
                if resets>1:raise JourneyFailed('window defence wiped twice')
            if any(e.type=='dungeon_complete' for e in evs):cleared=True;break
            snap=j.snap();state=snap.get('swarm') or {};phase=state.get('phase')
            if phase and (not phases or phases[-1]!=phase):phases.append(phase);j.log('  swarm: '+phase)
            if snap['self'].get('downed'):time.sleep(.4);continue
            hostiles=j._hostiles(100)
            if hostiles:
                if j.engage(hostiles[0],timeout_s=90):kills+=1
                continue
            if phase in ('west_ready','east_ready','west_work','east_work'):
                west=phase.startswith('west');x,z=(53.5,65.5) if west else (75.5,65.5)
                if math.hypot(j.me()['x']-x,j.me()['z']-z)>1.5:
                    if not j.walk_to(x,z,y=67,radius=1.3,label='guard '+phase).ok:raise JourneyFailed('cannot reach work circle')
                j.stop()
                if phase.endswith('ready'):
                    j.rest();j.gw.call('landmark',landmarkId='swarm_west' if west else 'swarm_east');time.sleep(.3)
                else:time.sleep(.35)
            else:time.sleep(.2)
        if not cleared:raise JourneyFailed('window defence did not clear: '+str(j.snap().get('swarm')))
        if kills<9:raise JourneyFailed('expected eight adds and a hive; saw '+str(kills))
        return f'two windows, eight adds and hive; {kills} real kills; {resets} resets; {time.monotonic()-start:.1f}s'
    finally:
        j.stop();j.gw.call('enter',timeout_s=40,zone='overworld');time.sleep(2)

def steps(j,roads,choice='short_pulses'):
    from .journey import FROM_ASSEMBLY, WINDREACH_CRYSTAL, JourneyFailed
    baseline={};prior={};paths={k:[tuple(p) for p in v] for k,v in roads.items()}
    outer=sum([paths[k] for k in ('coast','aid','edge','scarPass','scarSurvey','pillars')],[])
    outer_back=sum([list(reversed(paths[k])) for k in ('pillars','scarSurvey','scarPass','edge','aid','coast')],[])
    inner=paths['ring']+paths['ante'];inner_back=list(reversed(paths['ante']))+list(reversed(paths['ring']))
    ids={q for q,*_ in STORIES}
    def old_rows():return {q['id']:(q.get('state'),q.get('have'),q.get('choiceId')) for q in j.snap().get('quests',[]) if q['id'] not in ids and q['id'].startswith(('act4_','act5_','life_')) and q['state']!='available'}
    def check_start():
        if j.quest_state('act5_vexar_report')!='completed':raise JourneyFailed('requires the saved Vexar report')
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
        if j.me()['coins']-baseline['coins']!=800:raise JourneyFailed('incorrect rewards')
        return 'five saved quests, 800 coins once, all prior records unchanged'
    result=[('verify','continue saved Vexar report',check_start),('travel','return from last visit if needed',home),('equipment','ordinary guild sword and shield',kit)]
    for q,giver,dest in STORIES:
        result.append(('quest','accept '+q,lambda q=q,n=giver:j.quest(n,q,'accept',via=[(182,116),(201,115)] if q=='act5_swarm_briefing' else None)))
        if q=='act5_swarm_briefing':result.append(('walk','confirm retreat at the black pillars',lambda:walk([(201,115),(182,115),(182,149)]+outer,'southern road to Ivara')))
        if q=='act5_swarm_route':result.append(('walk','bring retreat instructions to Tovan',lambda:walk(inner,'inner approach')))
        if q=='act5_swarm':result.append(('combat','guard both windows and clear the hive',lambda:protect_windows(j)))
        if q=='act5_swarm_readings':result.append(('walk','return the signed readings',lambda:walk(inner_back,'return to Ivara')))
        if q=='act5_swarm_report':result.append(('walk','bring the crew report home',lambda:walk(outer_back+[(182,149),(182,115),(201,115)],'return to Kael')))
        result.append(('quest','complete '+q,lambda q=q,n=dest:j.quest(n,q,'complete')))
    result.append(('verify','rejoin and check saved records',verify))
    return result
