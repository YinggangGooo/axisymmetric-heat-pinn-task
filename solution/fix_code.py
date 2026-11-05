from pathlib import Path

path = Path('/app/baseline_models.py')
source = path.read_text(encoding='utf-8')
changes = {
    'return derivative(dr, r) + derivative(dz, z)':
        'return dr/r + derivative(dr, r) + derivative(dz, z)',
    'return physics["H"] * derivative(temperature, z) + (physics["T_top"] - physics["T_bottom"])':
        'return temperature - physics["T_top"]',
}
for old, new in changes.items():
    if source.count(old) != 1:
        raise SystemExit('Reference repair expected exactly one matching expression')
    source = source.replace(old, new, 1)
path.write_text(source, encoding='utf-8')
path = Path('/app/error_metrics.py')
source = path.read_text(encoding='utf-8')
old = 'return np.abs(prediction - reference) / 80.0 * 100.0'
new = 'return np.abs(prediction - reference) / np.abs(reference) * 100.0'
if source.count(old) != 1:
    raise SystemExit('Reference repair expected exactly one error normalization')
path.write_text(source.replace(old, new, 1), encoding='utf-8')
