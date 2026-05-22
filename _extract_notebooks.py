import json

notebooks = [
    ('eeg_fundamentos_e_arquitetura.ipynb', 'extracted_notebook1.txt'),
    ('eegnet_motor_imagery.ipynb', 'extracted_notebook2.txt'),
    ('eegnet_mi_improvements.ipynb', 'extracted_notebook3.txt'),
]

for nb_name, out_name in notebooks:
    nb = json.load(open(nb_name, 'r', encoding='utf-8'))
    total = len(nb['cells'])
    with open(out_name, 'w', encoding='utf-8') as out:
        for i, c in enumerate(nb['cells']):
            cell_type = c['cell_type'].upper()
            source = ''.join(c.get('source', []))
            out.write(f'=== [{cell_type}] Cell {i+1}/{total} ===\n')
            out.write(source)
            out.write('\n\n')
            for o in c.get('outputs', []):
                otype = o.get('output_type', '')
                if otype == 'stream':
                    out.write('--- OUTPUT ---\n')
                    out.write(''.join(o.get('text', [])))
                    out.write('\n')
                elif otype in ('execute_result', 'display_data'):
                    data = o.get('data', {})
                    if 'text/plain' in data:
                        out.write('--- OUTPUT ---\n')
                        out.write(''.join(data['text/plain']))
                        out.write('\n')
                    if 'image/png' in data:
                        out.write('[IMAGE/PLOT OUTPUT]\n')
                elif otype == 'error':
                    out.write('--- ERROR ---\n')
                    out.write('\n'.join(o.get('traceback', [])))
                    out.write('\n')
    print(f'OK: {nb_name} -> {out_name} ({total} cells)')
