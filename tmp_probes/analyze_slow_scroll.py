"""Analyze today's diagnostics JSONL for the few-Hz-scroll regression."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'arrayscope-diagnostics-20260705-121518.jsonl'
rows = [json.loads(l) for l in open(path)]

def g(r, *ks):
    v = r.get('diagnostics')
    for k in ks:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            return None
    return v

evs = {}
for r in rows:
    evs[r.get('event')] = evs.get(r.get('event'), 0) + 1
print('rows:', len(rows), 'events:', evs)
print('time span:', rows[0].get('recorded_at'), '->', rows[-1].get('recorded_at'))

print('\nseq | t | render_sync | disp_commit | init_commit | cache_resolve | stalls | staleids | pend | evalg | sid')
for r in rows:
    print(r.get('sequence'), str(r.get('recorded_at', ''))[11:23],
          g(r, 'render_timing', 'last_render_sync_ms'),
          g(r, 'render_timing', 'last_display_commit_ms'),
          g(r, 'montage_timing', 'last_initial_commit_ms'),
          g(r, 'montage_timing', 'last_cache_resolve_ms'),
          g(r, 'montage', 'stall_repairs'),
          g(r, 'montage', 'backend_stale_identities'),
          g(r, 'montage', 'pending_tiles'),
          g(r, 'montage', 'lifecycle_evaluating'),
          g(r, 'montage', 'session_id'), sep=' | ')
