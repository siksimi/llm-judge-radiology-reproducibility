"""r2_5_source_characteristics.py - Supplementary Table S15 (R2-5).
Re-runs the dataset builder's parse/filter/sample steps on the Open-i XML files and describes the eligible pool and the sample.
Run from anywhere: python3 r2_5_source_characteristics.py <YH2 repo root> <out.csv>"""
import sys, random, csv, statistics as st
from pathlib import Path
import xml.etree.ElementTree as ET
root = Path(sys.argv[1]); sys.path.insert(0, str(root / 'src'))
import dataset_builder as db
xmls = sorted((root / 'data/openi_reports').rglob('*.xml'))
allp = [r for r in (db.parse_iu_xml(x) for x in xmls) if r]
parsed = [r for r in allp if db._is_usable(r)]
empty = sum(1 for r in allp if not r['findings'] or r['findings'].lower() in {'none.', 'none', 'xxxx.', ''})
sample = random.Random("20260622|sample").sample(parsed, 100)
meta = {row['uid'] for row in csv.DictReader(open(root / 'data/metadata.csv'))}
assert {r['uid'] for r in sample} == meta, 'sample does not reproduce data/metadata.csv'
norm = {}
for x in xmls:
    r = ET.parse(x).getroot(); u = r.find('.//uId')
    if u is not None:
        norm['uId:' + u.get('id')] = 'normal' in [(m.text or '').strip().lower() for m in r.iter('major')]
def desc(R):
    q = lambda v: f"{st.median(v):.0f} ({st.quantiles(v, n=4)[0]:.0f}–{st.quantiles(v, n=4)[2]:.0f})"
    c = lambda k: (k, len(R))
    return {
        'words_findings_impression': q([len(r['findings'].split()) + len(r['impression'].split()) for r in R]),
        'words_findings': q([len(r['findings'].split()) for r in R]),
        'impression_present': c(sum(1 for r in R if r['impression'])),
        'indication_present': c(sum(1 for r in R if r['indication'])),
        'comparison_recorded': c(sum(1 for r in R if r['comparison'] and r['comparison'].lower().strip('. ') not in ('none', ''))),
        'xxxx_masking': c(sum(1 for r in R if 'xxxx' in (r['findings'] + ' ' + r['impression']).lower())),
        'mesh_major_normal': c(sum(1 for r in R if norm.get(r['uid']))),
    }
P, S = desc(parsed), desc(sample)
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['characteristic', 'pool', 'sample'])
    w.writerow(['n_xml', len(xmls), '']); w.writerow(['excluded_empty_or_placeholder', empty, '']); w.writerow(['excluded_short', len(allp) - empty - len(parsed), ''])
    w.writerow(['n', len(parsed), len(sample)])
    for k in P: w.writerow([k, P[k], S[k]])
print(open(sys.argv[2]).read())
