"""Generate one shareable Python file with the exact benchmark workload."""
import base64
import hashlib
import io
from pathlib import Path
import zipfile


def package(destination):
    root = Path(__file__).resolve().parents[1]
    modules = ("__init__", "config", "regions", "neurons", "network", "body", "world",
               "mujoco_body", "sim", "motor", "sensory", "metrics", "learning")
    files = [root / (name + ".py") for name in modules]
    files += [root / "assets/hexapod.xml", root / "assets/hexapod_mk1.xml"]
    files += sorted((root / "assets/ladybird").glob("*.obj"))
    files += sorted((root / "assets/mk1").glob("*.obj"))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(files):
            name = "veguinha_mini/" + path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, (2026, 10, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    payload = stream.getvalue()
    source = (root / "tools/portable_benchmark.py").read_text()
    source = source.replace('PAYLOAD = ""  # Replaced by package_benchmark.py, not downloaded at runtime.',
                            "PAYLOAD = " + repr(base64.b85encode(payload).decode("ascii")))
    source = source.replace('PAYLOAD_SHA256 = ""', "PAYLOAD_SHA256 = " + repr(hashlib.sha256(payload).hexdigest()))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    print(f"{destination}: {destination.stat().st_size} bytes; {len(files)} arquivos embutidos")
    return destination


if __name__ == "__main__":
    package(Path.home() / "storage/downloads/veguinha_benchmark.py")
