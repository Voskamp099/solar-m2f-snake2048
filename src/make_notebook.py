from pathlib import Path
import json

root = Path(__file__).resolve().parent
cells = [dict(cell_type='markdown', metadata={}, source=[
    '# Solar Mask2Former R50 best checkpoint — inference only\n',
    'Checkpoint iteration 36999; original 2048 resolution; threshold 0.5.\n',
    'No training or optimizer updates. Validate the saved checkpoint, then predict all 180 test images.\n'])]
for name in ['setup_kaggle.py', 'predict_kaggle.py']:
    source = (root / name).read_text()
    compile(source, name, 'exec')
    cells.append(dict(cell_type='code', execution_count=None, metadata={},
                      outputs=[], source=source.splitlines(keepends=True)))
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.13')), nbformat=4,nbformat_minor=5)
path = root / 'Solar_M2F_Best_Inference.ipynb'
path.write_text(json.dumps(nb, indent=2))
print(path)
