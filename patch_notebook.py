import json

with open('allwires_v1.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

new_source = []
for line in d['cells'][3]['source']:
    if '_backup = pd.read_csv' in line: continue
    if '_backup.set_index' in line: continue
    new_source.append(line)

d['cells'][3]['source'] = new_source

with open('allwires_v1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
