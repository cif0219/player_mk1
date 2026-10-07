"""Act V opening: ordinary travel, three-city checks and a real solo relay fight."""
from .city_life import rejoin, record_baseline

STORIES = [
    ('act5_council','seraine_kael','aldris_vane',None),
    ('act5_warning','aldris_vane','aldris_vane','resonance_warning'),
    ('act5_supply_visit','aldris_vane','brondt_ashvein',None),
    ('act5_supply','brondt_ashvein','opaline_vess','resonance_supply'),
    ('act5_method_visit','opaline_vess','archscholar_selara',None),
    ('act5_method','archscholar_selara','windcaller_ysolde','resonance_method'),
    ('act5_core_briefing','windcaller_ysolde','recorder_tovan',None),
    ('act5_resonant_core','recorder_tovan','recorder_tovan',None),
    ('act5_core_calibration','recorder_tovan','recorder_tovan','resonance_calibration'),
    ('act5_resonance_report','recorder_tovan','seraine_kael',None),
]
STATIONS={
    'resonance_warning':(124,74,None,['send','reply','relief']),
    'resonance_supply':(74,29,25,['reserve','reduce','stop']),
    'resonance_method':(-1,-194,151,['independent','conditions','correction']),
    'resonance_calibration':(442,445,95,['three','settle','receiver']),
}

def practice(j,q,station,stations=None):
    from .journey import JourneyFailed
    x,z,y,answers=(STATIONS if stations is None else stations)[station]
    if y is None: y=round(j.me()['y'])
    if not j.walk_to(x+.5,z+1.5,y=y,radius=.45,label=station).ok:raise JourneyFailed('cannot reach '+station)
    j.settle()
    def open_node():
        j.drain_events();j.gw.call('landmark',landmarkId=station)
        return j.wait_event('npc_dialog',lambda d:str(d.get('nodeId','')).startswith('life:'+station+':'))
    def choose(node,option):
        j.gw.call('choose',entityId=node['entityId'],nodeId=node['nodeId'],optionId=option)
        return j.wait_event('npc_dialog',lambda d:str(d.get('nodeId','')).startswith('life:'+station+':'))
    node=open_node()
    for i,answer in enumerate(answers):
        if node['nodeId'].split(':')[2]!=str(i):raise JourneyFailed('unexpected saved step')
        node=choose(node,'reconsider')
        if node['nodeId'].split(':')[2]!=str(i) or '.feedback.' not in node['messageKey']:raise JourneyFailed('wrong answer advanced progress')
        node=choose(node,answer)
        if i==0:
            j.gw.fire('close_dialog');rejoin(j)
            if j.quest_entry(q).get('have')!=1:raise JourneyFailed('partial check lost')
            node=open_node()
    j.gw.fire('close_dialog')
    j.gw.wait_snapshot(lambda d:any(r['id']==q and r['state']=='ready' for r in d.get('quests',[])),timeout_s=8)
    return 'three checks; wrong choices rejected; partial progress survived reconnect'

def steps(j,roads,choice='short_pulses'):
    from .journey import ROUTES, KEEP, TO_ASSEMBLY, FROM_ASSEMBLY, DEEP_LIGHT_CRYSTAL, WINDREACH_CRYSTAL, JourneyFailed
    baseline={};prior={}
    paths={k:[tuple(p) for p in v] for k,v in roads.items()}
    def old_rows():
        return {q['id']:(q.get('state'),q.get('have'),q.get('choiceId')) for q in j.snap().get('quests',[]) if q['id'].startswith('act4_') or q['id'].startswith('life_')}
    def check_start():
        if j.quest_state('act4_threshold_report')!='completed':raise JourneyFailed('requires the completed threshold report')
        plan=j.quest_entry('act4_scar_plan')
        if plan and plan.get('choiceId') is not None and plan['choiceId']!=choice:raise JourneyFailed('prior plan changed')
        prior.update(old_rows());return record_baseline(j,baseline)
    def kit():
        # A saved survey-only character may not yet have claimed ordinary guild weapons.
        if j.quest_state('guild_armament')!='completed':
            if j.quest_state('guild_orientation')!='completed':
                if j.quest_state('guild_orientation') not in ('active','ready'):j.quest('guild_mira','guild_orientation','accept',via=[(201,112),(201,102),(194,94)])
                j.quest('recruit_ren','guild_orientation','complete')
            if j.quest_state('guild_armament') not in ('active','ready'):j.quest('recruit_ren','guild_armament','accept')
            j.quest('guild_mira','guild_armament','complete')
        return j.equip_items([('guild_sword','left'),('guild_shield','right')])
    def descend():
        for x,z,y in [(58.5,100.5,64),(58.5,56,25),(60,36,25)]:
            if not j.walk_to(x,z,y=y,radius=1 if y==64 else 2,label='Delver descent').ok:raise JourneyFailed('mine stairs failed')
        return 'walked into Duskhollow'
    def to_wind():
        j.travel(DEEP_LIGHT_CRYSTAL,'luminara')
        route=ROUTES['city_to_ironvein']+paths['woods'][1:]+[(-56,22)]+paths['north']+paths['lower'][1:]+[(-4,-148)]+paths['upper']+[(8,-176),(8,-190),(8,-203)]
        if not j.walk_route(route,'mountain road to Windreach'):raise JourneyFailed('mountain road failed')
        return 'climbed the complete pass; Windreach crystal attuned on arrival'
    def to_core():
        j.walk_route([(0,-190),(8,-190)],'return to Galeheart')
        j.travel(WINDREACH_CRYSTAL,'luminara')
        route=[(152,116),(182,116),(182,149)]+sum([paths[k] for k in ('coast','aid','edge','scarPass','scarSurvey','pillars','ring','ante')],[])
        if not j.walk_route(route,'Crucible approach'):raise JourneyFailed('southern approach failed')
        return 'three-city records carried along the surveyed route'
    def verify():
        rejoin(j)
        if any(j.quest_state(q)!='completed' for q,*_ in STORIES):raise JourneyFailed('completion lost')
        if old_rows()!=prior:raise JourneyFailed('older story or side progress changed')
        if j.me()['coins']-baseline['coins']!=1100:raise JourneyFailed('wrong reward total')
        return '10 quests saved; 1100 coins once; earlier stories unchanged; core defeated through ordinary combat'
    result=[('verify','continue the saved threshold report',check_start),('equipment','claim and equip ordinary guild weapons',kit)]
    for q,giver,dest,station in STORIES:
        via=[(201,102),(201,115)] if q=='act5_council' else [(8,-196),(8,-203)] if q=='act5_method' else None
        result.append(('quest','accept '+q,lambda q=q,n=giver,v=via:j.quest(n,q,'accept',via=v)))
        if q=='act5_supply_visit':
            result.extend([('walk','west causeway to minehead',lambda:j.walk_route(FROM_ASSEMBLY+ROUTES['city_to_ironvein']+[(60,108),(58,104)],'minehead')),('walk','descend to Duskhollow',descend)])
        if q=='act5_method_visit':result.append(('travel','crystal home, then climb to Windreach',to_wind))
        if q=='act5_method':result.append(('walk','leave the Athenaeum for the Sanctum',lambda:j.walk_route([(8,-203),(8,-196),(-3,-193)],'Sanctum')))
        if q=='act5_core_briefing':result.append(('travel','return home, then walk to the antechamber',to_core))
        if station:result.append(('survey',station,lambda q=q,s=station:practice(j,q,s)))
        if q=='act5_resonant_core':result.append(('combat','disable the Resonant Core solo',lambda:j.clear_instance('story_resonant_core','resonant_core')))
        via=KEEP+TO_ASSEMBLY if q=='act5_council' else None
        if q=='act5_resonance_report':via=sum([list(reversed(paths[k])) for k in ('ante','ring','pillars','scarSurvey','scarPass','edge','aid','coast')],[])+[(182,149),(182,115),(201,115)]
        result.append(('quest','complete '+q,lambda q=q,n=dest,v=via:j.quest(n,q,'complete',via=v)))
    result.append(('verify','saved progress and rewards after returning',verify))
    return result
