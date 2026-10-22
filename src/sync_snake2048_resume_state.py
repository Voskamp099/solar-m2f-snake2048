from pathlib import Path
import json
P=Path(__file__).resolve().parent;O=P/'snake2048_resume_20261018'
live=json.loads((O/'live_progress.json').read_text());events=live['events'];last=events[-1]
probes=[e['data'] for e in events if e['marker']=='RESUME2048_PREFLIGHT_PASS']
complete=[e['data'] for e in events if e['marker']=='RESUME2048_ARM_COMPLETE']
rounds=[e['data'] for e in events if e['marker']=='RESUME2048_ROUND_COMPLETE']
for name in ['cycle_monitor_state.json','resume_state.json']:
 f=P/name;d=json.loads(f.read_text());d.update(status='full2048_continuation_running',phase='full2048_continuation_running',gpu_session_status='running',active_experiment='snake2048_resume_20261018',updated_at_utc=live['observed_at']);d['full2048_continuation'].update(status='running',gpu_status='running',preflights=probes,completed_arms=complete,completed_rounds=rounds,last_progress=last,updated_at_utc=live['observed_at'],live_log=str(O/'live_gpu_stdout.txt'));f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
f=P/'experiment_index_20261017.json';d=json.loads(f.read_text());d.update(status='full2048_continuation_running',gpu='running',updated_at_utc=live['observed_at'])
for e in d['experiments']:
 if e['id']=='E09_full2048_matched_continuation':e.update(status='running',preflights=probes,completed_arms=complete,completed_rounds=rounds,last_progress=last)
f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
print(json.dumps(dict(last_marker=last['marker'],last_arm=last['data'].get('arm'),completed_arms=len(complete),completed_rounds=len(rounds))))
