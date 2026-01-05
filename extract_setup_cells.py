import json
import os

# Extract setup cells from quantitative analysis notebook
notebooks_to_check = [
    ('Quantitative_SAM_Improvement_Analysis_colab.ipynb', 15),
    ('Hierarchical_Decision_and_Agentic_System_(YOLO_+_SAM_3_+_Agent).ipynb', 10)
]

output = []

for nb_name, max_cells in notebooks_to_check:
    path = os.path.join(r'd:\SHEZAN\AI\scrpv', nb_name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        output.append('\n' + '='*80 + '\n')
        output.append(f'=== {nb_name} ===\n')
        output.append('='*80 + '\n')
        
        for i, cell in enumerate(nb['cells'][:max_cells]):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                # Skip cells that already have run_sam_rescue (already extracted)
                if 'run_sam_rescue' not in source:
                    output.append(f'\n--- CELL {i} ---\n')
                    output.append(source)
                    output.append('\n')

with open(r'd:\SHEZAN\AI\scrpv\notebook_setup_cells.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(output))

print('Saved to notebook_setup_cells.txt')
