import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def completed_job(folder, expected_fingerprint):
    folder = Path(folder)
    marker = folder / "result.json"
    if not marker.exists():
        return False
    try:
        result = read_json(marker)
        if (
            result.get("status") != "complete"
            or result.get("fingerprint") != expected_fingerprint
            or not result.get("artifact_sha256")
        ):
            return False
        for relative, digest in result["artifact_sha256"].items():
            path = (folder / relative).resolve()
            if (
                not path.is_relative_to(folder.resolve())
                or not path.is_file()
                or sha256(path) != digest
            ):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False
