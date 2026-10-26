"""Portable filesystem configuration; importing does not create directories."""
import json, os
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
def load_paths(config=None):
    path=Path(config or os.environ.get('SIVR_PATHS',str(REPO/'configs/paths.example.json'))).expanduser().resolve()
    values=json.loads(path.read_text(encoding='utf-8'))
    defaults={'replication_root':str(REPO/'runs/replication'),'replication_cache':str(REPO/'data/cache/replication'),'development_root':str(REPO/'runs/development'),'development_base':str(REPO/'data/development'),'development_cache':str(REPO/'data/cache/development'),'data_root':str(REPO/'data/GOD-Wiki'),'backbone_root':str(REPO/'third_party/bravl')}
    result={}
    for key,fallback in defaults.items():
        value=Path(os.path.expandvars(str(values.get(key,fallback)))).expanduser()
        result[key]=(value if value.is_absolute() else path.parent/value).resolve()
    return result
