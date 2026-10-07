"""Crossroads: real roads, seven witnessed activities, final report; never production setup commands."""
import time
from .city_life import rejoin, record_baseline
STORIES=[
 ('act5_accord_invitation','aldris_vane','keepmaster_dariun',None,100),
 ('act5_accord_water','keepmaster_dariun','wellwright_eska','accord_water',80),
 ('act5_accord_relief','wellwright_eska','host_mera','accord_relief',80),
 ('act5_accord_load','host_mera','cartwright_rennel','accord_load',100),
 ('act5_accord_records','cartwright_rennel','registrar_halea','accord_records',100),
 ('act5_accord_terms','registrar_halea','keepmaster_dariun','accord_council',120),
 ('act5_accord_sign','keepmaster_dariun','keepmaster_dariun','accord_sign',160),
 ('act5_accord_handoff','keepmaster_dariun','courier_jori','accord_handoff',100),
 ('act5_accord_report','courier_jori','seraine_kael',None,180),
]
STATIONS={
 'accord_water':(-20,151,2),'accord_relief':(-42,151,2),'accord_load':(-45,141,2),
 'accord_records':(-13,130,2),'accord_council':(-30,127,3),'accord_sign':(-30,132,2),'accord_handoff':(-13,143,2),
}

def practice(j,q,station):
 from .journey import JourneyFailed
 if j.quest_state(q) in ('ready','completed'):return 'previously witnessed'
 x,z,steps=STATIONS[station]
 # The public reading-room table is approached from its open south side.
 if not j.walk_to(x+.5,z+1.5,y=71,radius=.45,label=station).ok:raise JourneyFailed('cannot reach '+station)
 j.settle()
 def open_node():
  j.drain_events();j.gw.call('landmark',landmarkId=station)
  return j.wait_event('npc_dialog',lambda d:str(d.get('nodeId','')).startswith('life:'+station+':'))
 def choose(node):
  j.gw.call('choose',entityId=node['entityId'],nodeId=node['nodeId'],optionId='record')
  return j.wait_event('npc_dialog',lambda d:str(d.get('nodeId','')).startswith('life:'+station+':'))
 node=open_node()
 for i in range(int(j.quest_entry(q).get('have',0)),steps):
  if node['nodeId'].split(':')[2]!=str(i):raise JourneyFailed('unexpected saved step')
  node=choose(node)
  if i==0:
   j.gw.fire('close_dialog');rejoin(j)
   if j.quest_entry(q).get('have')!=1:raise JourneyFailed('partial action lost')
   node=open_node()
 j.gw.fire('close_dialog')
 j.gw.wait_snapshot(lambda d:any(r['id']==q and r['state']=='ready' for r in d.get('quests',[])),timeout_s=8)
 return f'{steps} actions; partial progress survived reconnect'

def steps(j,roads,choice='short_pulses'):
 from .journey import FROM_ASSEMBLY,TO_ASSEMBLY,KEEP,ROUTES,ACCORD_ECHOES,JourneyFailed
 baseline={};prior={};pending_legacy=j.quest_state('kael_training')!='completed';paths={k:[tuple(p) for p in v] for k,v in roads.items()};ids={q for q,*_ in STORIES}
 def walk(route,label):
  if not j.walk_route(route,label):raise JourneyFailed(label+' failed')
  return label
 def old_rows():return {q['id']:(q.get('state'),q.get('have'),q.get('choiceId')) for q in j.snap().get('quests',[]) if q['id'] not in ids and q['state']!='available'}
 def quest(n,q,action,via=None):
  state=j.quest_state(q)
  if state=='completed' or (action=='accept' and state in ('active','ready')):return 'saved '+q
  return j.quest(n,q,action,via=via)
 def home():
  if j.quest_state('act5_maw_report')!='completed':raise JourneyFailed('requires saved Maw report')
  plan=j.quest_entry('act4_scar_plan')
  if plan and plan.get('choiceId') not in (None,choice):raise JourneyFailed('prior plan changed')
  if j.me()['z']<95 and 100<j.me()['x']<150:walk(FROM_ASSEMBLY,'leave the Assembly')
  return 'saved plan retained'
 def kit():
  if j.quest_state('guild_armament')!='completed':
   quest('guild_mira','guild_orientation','accept',via=KEEP+[(201,102),(194,94)])
   quest('recruit_ren','guild_orientation','complete')
   quest('recruit_ren','guild_armament','accept');quest('guild_mira','guild_armament','complete')
   walk([(201,102),(201,115)],'leave the guild')
  return j.equip_items([('guild_sword','left'),('guild_shield','right')])
 def record():
  note=record_baseline(j,baseline)
  prior.update(old_rows());baseline['remaining']=sum(coins for q,_,_,_,coins in STORIES if j.quest_state(q)!='completed' and not(pending_legacy and q=='act5_accord_report'))
  return note
 def verify():
  rejoin(j)
  if any(j.quest_state(q)!=('active' if pending_legacy and q=='act5_accord_report' else 'completed') for q,*_ in STORIES):raise JourneyFailed('completion or pending gate lost')
  if not pending_legacy and j.quest_state('accord_home')!='completed':raise JourneyFailed('shared-watch record missing')
  if old_rows()!=prior:raise JourneyFailed('previous records changed: '+str({k:(prior.get(k),old_rows().get(k)) for k in prior.keys()|old_rows().keys() if prior.get(k)!=old_rows().get(k)}))
  if j.me()['coins']-baseline['coins']!=baseline['remaining']:raise JourneyFailed('incorrect rewards')
  return ('eight completed quests; final report correctly pending missing early story; 840 coins' if pending_legacy else 'nine completed quests; 1020 coins')+'; seven reconnects; 15 actions; prior records preserved'
 def final_report():
  if not pending_legacy:return quest('seraine_kael','act5_accord_report','complete')
  j.gw.fire('snapshot');time.sleep(.3)
  if j.quest_state('act5_accord_report')!='active':raise JourneyFailed('legacy prerequisite gate bypassed')
  npc=j.approach_npc('seraine_kael');j.settle();j.drain_events();j.gw.call('talk',entityId=npc['entityId'])
  menu=j.wait_event('npc_dialog',lambda d:bool(d.get('options')))
  j.gw.call('choose',entityId=npc['entityId'],nodeId=menu['nodeId'],optionId='act5_accord_report')
  node=j.wait_event('npc_dialog',lambda d:str(d.get('nodeId','')).startswith('quest:act5_accord_report'))
  if any(o.get('id')=='complete' for o in node.get('options') or []):raise JourneyFailed('final reward incorrectly offered')
  j.gw.fire('close_dialog');return 'final report explains missing records; no reward or completion offered'
 result=[('verify','continue saved Maw report',home)]
 if not pending_legacy and j.quest_state('accord_home')!='completed':
  result.append(('equipment','ordinary guild weapons for historical echoes',kit))
  result.append(('walk','return to Kael for unfinished historical records',lambda:walk(KEEP,'Keep gate')))
  result.append(('quest','accept Seven Echoes',lambda:quest('seraine_kael','accord_invitation','accept')))
  result.append(('quest','Oswin receives the investigation',lambda:quest('warden_oswin','accord_invitation','complete',via=[(182,116),(152,116),(134,116)])))
  for q,zone,boss in ACCORD_ECHOES:
   result.extend([
    ('quest','accept '+q,lambda q=q:quest('warden_oswin',q,'accept')),
    ('combat','investigate '+zone,lambda q=q,z=zone,b=boss:'saved echo' if j.quest_state(q) in ('ready','completed') else j.clear_instance(z,b)),
    ('quest','complete '+q,lambda q=q:quest('warden_oswin',q,'complete'))])
  for q,giver,dest,route in [
   ('accord_assembly','warden_oswin','archivist_quill',[(152,116),(152,98),(167,98),(167,94)]),
   ('accord_beacon','archivist_quill','sister_amalthe',[(167,98),(152,98)]),
   ('accord_home','sister_amalthe','seraine_kael',[(152,98)]+KEEP)]:
   result.extend([('quest','accept '+q,lambda q=q,n=giver:quest(n,q,'accept')),('quest','complete '+q,lambda q=q,n=dest,v=route:quest(n,q,'complete',via=v))])
 result.append(('verify','baseline after both historical branches',record))
 for q,giver,dest,station,_ in STORIES:
  result.append(('quest','accept '+q,lambda q=q,n=giver:quest(n,q,'accept',via=[(201,115),(182,116),(128,116)]+TO_ASSEMBLY if q=='act5_accord_invitation' else None)))
  if q=='act5_accord_invitation':result.append(('walk','west causeway and east gate of Crossroads',lambda:walk(FROM_ASSEMBLY+ROUTES['city_to_ironvein']+paths['crossroadsEast']+[(-29,145)],'Crossroads east road')))
  if q=='act5_accord_records':result.append(('walk','open reading room',lambda:walk([(-30,145),(-12,140),(-12,133)],'reading room entrance')))
  if q=='act5_accord_terms':result.append(('walk','meeting court',lambda:walk([(-20,138),(-30,138),(-30,130)],'meeting court')))
  if station:result.append(('witness',station,lambda q=q,s=station:practice(j,q,s)))
  if q=='act5_accord_report':result.append(('walk','north gate and existing woods road home',lambda:walk([(-20,145),(-30,145)]+paths['crossroadsNorth']+[(50,116),(60,116),(86,116),(128,116)]+KEEP,'Crossroads north return')))
  result.append(('quest','verify final report gate' if pending_legacy and q=='act5_accord_report' else 'complete '+q,final_report if q=='act5_accord_report' else lambda q=q,n=dest:quest(n,q,'complete')))
 result.append(('verify','final report survives reconnect',verify))
 return result
