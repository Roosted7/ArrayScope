"""Tail churn: why does backend_stale_identities flap at idle? And commit/draw deltas."""
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

print('seq | staleids | drawcnt | tp_draw | tp_req | coalesced | dirty | upserts | parked | presented | active | lodlvl')
prev = None
for r in rows:
    if r.get('sequence') is None:
        continue
    row = (r['sequence'],
           g(r, 'montage', 'backend_stale_identities'),
           g(r, 'montage', 'presentation_draw_count'),
           g(r, 'montage', 'tile_presentation_draw_count'),
           g(r, 'montage', 'tile_presentation_request_count'),
           g(r, 'montage_timing', 'coalesced_commits'),
           g(r, 'montage', 'dirty_payload_tiles'),
           g(r, 'montage', 'pending_payload_upserts'),
           g(r, 'montage', 'lifecycle_parked'),
           g(r, 'montage', 'lifecycle_presented'),
           g(r, 'montage', 'active'),
           g(r, 'montage', 'tile_lod_applied_level'))
    print(*row, sep=' | ')

# stall signature if any
for r in rows:
    sig = g(r, 'montage', 'last_stall_signature')
    if sig:
        print('STALL SIG @', r.get('sequence'), ':', json.dumps(sig)[:400])
        break
