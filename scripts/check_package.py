"""Read-only syntax and evaluator smoke checks for this snapshot."""
import ast
import json
import re
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
errors = []
for path in root.rglob('*.py'):
    try:
        ast.parse(path.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        errors.append([str(path.relative_to(root)), str(exc)])
passed = []
for path in sorted((root / 'tests/evaluation').glob('test_*.py')):
    for name, fn in runpy.run_path(str(path)).items():
        if name.startswith('test_') and callable(fn):
            fn()
            passed.append(f'{path.name}::{name}')
print(json.dumps({'syntax_errors': errors, 'evaluation_tests_passed': passed}, indent=2))
assert not errors
