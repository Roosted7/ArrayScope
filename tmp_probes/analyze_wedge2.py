"""Wrong-LOD wedge triage for the 2026-07-05 13:11-13:13 JSONL triptych."""
import json, sys

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]

def g(r, *ks):
    v = r.get('diagnostics')
    for k in ks:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            return None
    return v

print(f"== {path}: rows={len(rows)} span {rows[0].get('recorded_at','')[11:19]} -> {rows[-1].get('recorded_at','')[11:19]}")
print("seq | t | sid | staleids | idrej | stalls | dirty | upserts | pend | evalg | lodlvl | layer_reason | deferred")
prev = {}
for r in rows:
    if r.get('sequence') is None:
        continue
    m = lambda k: g(r, 'montage', k)
    row = dict(
        seq=r['sequence'], t=r.get('recorded_at', '')[11:23],
        sid=m('session_id'), stale=m('backend_stale_identities'),
        idrej=m('lifecycle_identity_rejections'), stalls=m('stall_repairs'),
        dirty=m('dirty_payload_tiles'), ups=m('pending_payload_upserts'),
        pend=m('pending_tiles'), ev=m('lifecycle_evaluating'),
        lvl=m('tile_lod_applied_level'), reason=str(m('tile_lod_reason'))[:44],
        parked=m('lifecycle_parked'),
    )
    # print only when something interesting changes or is nonzero
    interesting = (row['stale'] or row['idrej'] or row['stalls'] or row['dirty']
                   or row['ups'] or row['parked'])
    changed = any(row[k] != prev.get(k) for k in ('sid', 'stale', 'idrej', 'stalls', 'lvl', 'reason'))
    if interesting or changed:
        print(f"{row['seq']:>4} {row['t']} sid={row['sid']} stale={row['stale']} idrej={row['idrej']} "
              f"stalls={row['stalls']} dirty={row['dirty']} ups={row['ups']} pend={row['pend']} "
              f"evalg={row['ev']} parked={row['parked']} lvl={row['lvl']} | {row['reason']}")
    prev = row

last = rows[-1]
m = lambda k: g(last, 'montage', k)
print("\nTAIL:", {k: m(k) for k in (
    'session_id', 'backend_stale_identities', 'lifecycle_identity_rejections',
    'stall_repairs', 'dirty_payload_tiles', 'pending_payload_upserts',
    'pending_tiles', 'lifecycle_evaluating', 'lifecycle_parked',
    'tile_lod_applied_level', 'tile_lod_resident_tile_levels',
    'tile_lod_pending_materializations', 'tile_lod_reason',
    'last_stall_signature', 'retained_stage_decision', 'flush_pending',
    'final_commit_pending')})
