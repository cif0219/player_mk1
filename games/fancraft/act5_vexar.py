"""Act V second encounter: physical travel, source comparison and real phase combat."""
from .city_life import rejoin, record_baseline
from .act5_resonance import practice as resonance_practice

STORIES=[
 ('act5_vexar_briefing','seraine_kael','recorder_tovan',None),
 ('act5_vexar_record','recorder_tovan','recorder_tovan','vexar_record'),
 ('act5_vexar_compare','recorder_tovan','archscholar_selara','vexar_compare'),
 ('act5_vexar_return','archscholar_selara','recorder_tovan',None),
 ('act5_vexar','recorder_tovan','recorder_tovan',None),
 ('act5_vexar_release','recorder_tovan','recorder_tovan','vexar_release'),
 ('act5_vexar_report','recorder_tovan','seraine_kael',None),
]
STATIONS={
 'vexar_record':(454,445,95,['account','fragment','reading']),
 'vexar_compare':(-1,-190,151,['number','pending','bounded']),
 'vexar_release':(454,449,95,['order','observe','retain']),
}

def practice(j,q,station):
    # Share the tested interaction protocol, passing this chapter's immutable definitions.
    return resonance_practice(j,q,station,stations=STATIONS)

def steps(j,roads,choice='short_pulses'):
    from .journey import ROUTES, KEEP, FROM_ASSEMBLY, WINDREACH_CRYSTAL, JourneyFailed
    baseline={};prior={};paths={k:[tuple(p) for p in v] for k,v in roads.items()}
    outgoing=sum([paths[k] for k in ('coast','aid','edge','scarPass','scarSurvey','pillars','ring','ante')],[])
    incoming=sum([list(reversed(paths[k])) for k in ('ante','ring','pillars','scarSurvey','scarPass','edge','aid','coast')],[])
    ids={q for q,*_ in STORIES}
    def old_rows():
        return {q['id']:(q.get('state'),q.get('have'),q.get('choiceId')) for q in j.snap().get('quests',[]) if q['id'] not in ids and (q['id'].startswith(('act4_','act5_','life_')) and q['state']!='available')}
    def check_start():
        if j.quest_state('act5_resonance_report')!='completed':raise JourneyFailed('requires the saved resonance report')
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
    def to_wind():
        walk(incoming+[(182,149),(182,116),(152,116),(128,116)],'return to Luminara crystal')
        # Saved campaign characters use their attunement; a fresh local chapter fixture walks the pass.
        # The full route is deliberate: both fixture and saved characters exercise the same terrain.
        walk(ROUTES['city_to_ironvein']+paths['woods'][1:]+[(-56,22)]+paths['north']+paths['lower'][1:]+[(-4,-148)]+paths['upper']+[(8,-176),(8,-190),(0,-190)],'climb to Windreach comparison desk')
        return 'walked the surveyed return and the complete mountain pass'
    def return_front():
        walk([(8,-203),(8,-196),(8,-190)],'return to Galeheart crystal')
        j.travel(WINDREACH_CRYSTAL,'luminara')
        return walk([(152,116),(182,116),(182,149)]+outgoing,'bring procedure to the antechamber')
    def verify():
        rejoin(j)
        if any(j.quest_state(q)!='completed' for q,*_ in STORIES):raise JourneyFailed('completion lost')
        if old_rows()!=prior:raise JourneyFailed('previous progress changed')
        if j.me()['coins']-baseline['coins']!=900:raise JourneyFailed('wrong reward total')
        return '7 quests saved; 900 coins once; prior records unchanged; real solo Vexar combat'
    result=[('verify','continue the saved resonance report',check_start),('equipment','equip ordinary guild weapons',kit)]
    for q,giver,dest,station in STORIES:
        via=[(182,116),(201,115)] if q=='act5_vexar_briefing' and j.me()['x']<160 else [(201,102),(201,115)] if q=='act5_vexar_briefing' else None
        result.append(('quest','accept '+q,lambda q=q,n=giver,v=via:j.quest(n,q,'accept',via=v)))
        if q=='act5_vexar_briefing':result.append(('walk','walk the southern approach',lambda:walk([(201,115),(182,115),(182,149)]+outgoing,'southern approach')))
        if q=='act5_vexar_compare':result.append(('walk','carry the records to Windreach',to_wind))
        if q=='act5_vexar_return':result.append(('travel','return the checked procedure',return_front))
        if station:result.append(('survey',station,lambda q=q,s=station:practice(j,q,s)))
        if q=='act5_vexar':result.append(('combat','disperse Vexar solo',lambda:j.clear_instance('story_vexar','vexar')))
        via=[(8,-190),(8,-196),(8,-203)] if q=='act5_vexar_compare' else incoming+[(182,149),(182,115),(201,115)] if q=='act5_vexar_report' else None
        result.append(('quest','complete '+q,lambda q=q,n=dest,v=via:j.quest(n,q,'complete',via=v)))
    result.append(('verify','saved completion and old records',verify))
    return result
