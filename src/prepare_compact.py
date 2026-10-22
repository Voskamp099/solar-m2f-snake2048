from pathlib import Path
import json, zipfile

r=Path(__file__).resolve().parent
raw=Path('/Users/raelynn/Downloads/solar_project/MAGFiLO_1.0_Kaggle_2026')
records=json.loads((r/'bundle/validation_records.json').read_text())
names=sorted({x['file_name'] for x in records})
payload=r/'solar_payload.bin'
with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    with zipfile.ZipFile(r/'solar_m2f_best_inference_20261016.zip') as old:
        for name in old.namelist(): z.writestr(name,old.read(name))
    ann='MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    z.write(raw/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',ann)
    for name in names:
        z.write(raw/'train/train_images'/name,'MAGFiLO_1.0_Kaggle_2026/train/train_images/'+name)
    for p in sorted((raw/'test/test_images').glob('*.jpeg')):
        z.write(p,'MAGFiLO_1.0_Kaggle_2026/test/test_images/'+p.name)
with zipfile.ZipFile(r/'solar_compact_inference_20261016.zip','w',compression=zipfile.ZIP_STORED) as z:
    z.write(payload,payload.name)
print('compact_bytes',(r/'solar_compact_inference_20261016.zip').stat().st_size)
# Bootstrap a packed asset input; the prediction cell is unchanged.
setup=(r/'setup_kaggle.py').read_text()
old="""INPUTS = Path('/kaggle/input')
candidates = list(INPUTS.rglob('provenance.json'))
candidates = [p for p in candidates if (p.parent / 'model_best.pth').is_file()]
assert len(candidates) == 1, candidates
ASSETS = candidates[0].parent"""
new="""import zipfile
archives = list(Path('/kaggle/input').rglob('solar_payload.bin'))
assert len(archives) == 1, archives
ASSETS = Path('/kaggle/working/solar_assets')
ASSETS.mkdir(exist_ok=True)
with zipfile.ZipFile(archives[0]) as archive:
    archive.extractall(ASSETS)
INPUTS = ASSETS"""
assert old in setup
setup=setup.replace(old,new)
compile(setup,'compact_setup','exec')
nb=json.loads((r/'Solar_M2F_Best_Inference.ipynb').read_text())
nb['cells'][1]['source']=setup.splitlines(keepends=True)
(r/'Solar_M2F_Best_Compact.ipynb').write_text(json.dumps(nb,indent=2))
print('compact_notebook_ready')
