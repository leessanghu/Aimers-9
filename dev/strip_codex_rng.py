"""codex v20_905 아티팩트(phase_v12_submission.pkl, v20_905_submission.pkl) 안에
박힌 numpy RandomState/Generator(BitGenerator=PCG64 등)를 재귀적으로 제거한다.
build_v107_physhead.py에서 이미 검증된 방식(strip_rng) 그대로 재사용.
서버쪽 numpy 버전이 우리와 달라 pickle된 BitGenerator C-state를 못 읽는 문제 해결용.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import joblib

MODEL_DIR = 'submit/model'
sys.path.insert(0, os.path.abspath(MODEL_DIR))

_RNG = ('Generator', 'BitGenerator', 'RandomState', 'PCG64', 'MT19937', 'Philox', 'SFC64')


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 12 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, '__dict__'):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, dict):
        for k, v in list(obj.items()):
            if type(v).__name__ in _RNG:
                obj[k] = None
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def count_rng(obj, seen=None, depth=0, found=None):
    if seen is None:
        seen = set(); found = []
    if depth > 12 or id(obj) in seen:
        return found
    seen.add(id(obj))
    if type(obj).__name__ in _RNG:
        found.append(type(obj).__name__)
        return found
    if hasattr(obj, '__dict__'):
        for v in vars(obj).values():
            count_rng(v, seen, depth + 1, found)
    elif isinstance(obj, dict):
        for v in obj.values():
            count_rng(v, seen, depth + 1, found)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            count_rng(v, seen, depth + 1, found)
    return found


for fname in ('phase_v12_submission.pkl', 'v20_905_submission.pkl'):
    path = f'{MODEL_DIR}/{fname}'
    obj = joblib.load(path)
    before = count_rng(obj)
    print(f'{fname}: 제거 전 RNG객체 {len(before)}개 {set(before)}')
    strip_rng(obj)
    after = count_rng(obj)
    print(f'{fname}: 제거 후 RNG객체 {len(after)}개 {set(after)}')
    joblib.dump(obj, path)
    print(f'  재저장 완료: {path}')
