import json
import os

notebooks = [
    'Hierarchical_Decision_and_Agentic_System_(YOLO_+_SAM_3_+_Agent).ipynb',
    'Quantitative_SAM_Improvement_Analysis_colab.ipynb',
    'yolo11m_sam3_hybrid_detection.ipynb'
]

output = []
for nb_name in notebooks:
    path = os.path.join(r'd:\SHEZAN\AI\scrpv', nb_name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        output.append('\n\n' + '='*80 + '\n')
        output.append(f'=== {nb_name} ===\n')
        output.append('='*80 + '\n')
        
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'run_sam_rescue' in source or 'def run_sam' in source or 'sam_model(' in source:
                    output.append(f'\n--- CELL {i} ---\n')
                    output.append(source)
                    output.append('\n')

with open(r'd:\SHEZAN\AI\scrpv\notebook_sam_code_review.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(output))
print('Saved to notebook_sam_code_review.txt')
