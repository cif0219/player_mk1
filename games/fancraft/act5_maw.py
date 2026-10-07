"""Real movement, ordinary combat, explicit cancellation and three successful pressure tests."""
import time
from .city_life import rejoin, record_baseline
STORIES=[
 ('act5_maw_briefing','aldris_vane','recorder_tovan'),
 ('act5_maw_shift','recorder_tovan','roadwarden_ivara'),
 ('act5_maw_limits','roadwarden_ivara','recorder_tovan'),
 ('act5_maw','recorder_tovan','recorder_tovan'),
 ('act5_maw_watch','recorder_tovan','roadwarden_ivara'),
 ('act5_maw_report','roadwarden_ivara','aldris_vane'),
]
def rebind(j):
    from .journey import JourneyFailed
    j.drain_events();cursor=j.gw.event_cursor
    j.gw.call('enter',timeout_s=40,zone='story_wardens_maw');time.sleep(1.5)
    start=time.monotonic()
    def state():return j.snap().get('maw') or {}
    def walk(x,z):
        if not j.walk_to(x,z,y=67,radius=1,label='binding control approach').ok:raise JourneyFailed('cannot reach binding control')
        j.stop()
    def press(site):j.gw.call('landmark',landmarkId=site);time.sleep(.2)
    try:
        walk(64.5,78.5);press('maw_restart')
        if state().get('phase')!='manifestation':raise JourneyFailed('control loop did not restart')
        walk(52.5,64.5);press('maw_west');walk(76.5,64.5);press('maw_east')
        if state().get('pressure')!=44:raise JourneyFailed('bypass reading is wrong')
        target=j.entity(lambda e:e.get('mobId')=='wardens_maw' and e.get('alive'))
        if not target or not j.engage(target,timeout_s=150):raise JourneyFailed('manifestation combat failed')
        time.sleep(.2)
        if state().get('phase')!='binding':raise JourneyFailed('manifestation did not yield to binding work')
        if any(e.type=='dungeon_complete' for e in j.gw.events_since(cursor)):raise JourneyFailed('death alone incorrectly cleared room')
        for i,(x,z) in enumerate([(56.5,53.5),(72.5,53.5),(64.5,45.5)]):
            walk(x,z);site='maw_bind_'+str(i+1)
            if i==0:
                press(site);press(site)
                if state().get('last')!='aborted' or state().get('pressure')!=44:raise JourneyFailed('safe cancellation did not roll back load')
            press(site);walk(x,z+6)
            deadline=time.monotonic()+8
            while state().get('phase')=='proof' and time.monotonic()<deadline:time.sleep(.1)
            if state().get('sealed')!=i+1:raise JourneyFailed(f'binding {i+1} did not reply')
        evs=j.gw.events_since(cursor)
        if state().get('phase')!='stable' or state().get('pressure')!=8:raise JourneyFailed('final load is not stable')
        if sum(e.type=='dungeon_complete' for e in evs)!=1:raise JourneyFailed('completion missing or duplicated')
        if any(e.type=='instance_reset' for e in evs):raise JourneyFailed('encounter reset')
        j.report.kills['wardens_maw']=j.report.kills.get('wardens_maw',0)+1
        return f'actual manifestation combat; both bypasses; one cancellation; three safe proofs; load 84 to 44 to 8; {time.monotonic()-start:.1f}s'
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
        if j.quest_state('act5_primarch_report')!='completed':raise JourneyFailed('requires the saved Primarch report')
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
        if j.me()['coins']-baseline['coins']!=1000:raise JourneyFailed('incorrect rewards')
        return 'six saved quests, 1000 coins once, all prior records unchanged'
    result=[('verify','continue saved Primarch report',check_start),('travel','return from last visit if needed',home),('equipment','ordinary guild sword and shield',kit)]
    for q,giver,dest in STORIES:
        result.append(('quest','accept '+q,lambda q=q,n=giver:j.quest(n,q,'accept',via=[(128,116)]+TO_ASSEMBLY if q=='act5_maw_briefing' else None)))
        if q=='act5_maw_briefing':result.append(('walk','take the repair sequence to Tovan',lambda:walk(FROM_ASSEMBLY+[(182,116),(182,149)]+outer+inner,'southern road to Tovan')))
        if q=='act5_maw_shift':result.append(('walk','confirm the crew and relief',lambda:walk(inner_back,'return to Ivara')))
        if q=='act5_maw_limits':result.append(('walk','bring the checked stop procedure back',lambda:walk(inner,'inner approach')))
        if q=='act5_maw':result.append(('combat','restart, divert, disperse and prove three bindings',lambda:rebind(j)))
        if q=='act5_maw_watch':result.append(('walk','take the maintenance list to Ivara',lambda:walk(inner_back,'return to Ivara')))
        if q=='act5_maw_report':result.append(('walk','bring the readings home',lambda:walk(outer_back+[(182,149),(182,116),(128,116)]+TO_ASSEMBLY,'return to Vane')))
        result.append(('quest','complete '+q,lambda q=q,n=dest:j.quest(n,q,'complete')))
    result.append(('verify','rejoin and check saved records',verify))
    return result
