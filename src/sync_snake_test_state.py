from pathlib import Path
import json
P=Path(__file__).resolve().parent;O=P/'snake_best_submission_20261018'
live=json.loads((O/'live_progress.json').read_text());events=live['events']
plan=json.loads((O/'local_plan.json').read_text())
complete=next((e['data'] for e in reversed(events) if e['marker']=='SNAKE_TEST_COMPLETE'),None)
status='inference_complete_pending_download' if complete and complete['status']=='complete' else ('inference_error' if complete else 'inference_running')
plan.update(status=status,gpu_status='running',last_observed_at=live['observed_at'],last_event=events[-1] if events else None)
if complete:plan['inference_summary']=complete
(O/'local_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
for name in ['cycle_monitor_state.json','resume_state.json']:
    f=P/name;d=json.loads(f.read_text());d.update(status=status,phase=status,snake_best_submission=plan,gpu_session_status='running',updated_at_utc=live['observed_at'])
    f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
print(json.dumps(dict(status=status,last_marker=events[-1]['marker'] if events else None)))
