import json

with open('full_component_labeller.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace the syntax error
            if '"""Returns True=saved/continue, False=quit."""    img_orig' in line:
                line = line.replace('"""Returns True=saved/continue, False=quit."""    img_orig', '"""Returns True=saved/continue, False=quit."""\n    img_orig')
            new_source.append(line)
        cell['source'] = new_source

with open('full_component_labeller.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
