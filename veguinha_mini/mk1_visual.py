"""Build the approved mk-1 visual assembly without changing MuJoCo physics.

Requires trimesh, Pillow, Node.js and the local meshoptimizer dependency.
Run from the home root:
python -m veguinha_mini.mk1_visual
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .mk1_grouping import COLORS, classify, font, render


LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
SEGMENTS = ("coxa", "femur", "tibia")
FINAL_PALETTE = {
    "trunk": (75, 84, 92), "tail": (64, 72, 79),
    "head": (181, 143, 74), "mandible_L": (205, 162, 80),
    "mandible_R": (205, 162, 80),
}
SEGMENT_PALETTE = {"coxa": (66, 75, 82), "femur": (82, 91, 99),
                   "tibia": (95, 103, 109)}


def final_color(group: str, segment: str | None) -> tuple[int, int, int]:
    return SEGMENT_PALETTE[segment] if segment else FINAL_PALETTE[group]


def load_parts(source: Path) -> list[dict]:
    scene = trimesh.load(source, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        vertices = trimesh.transform_points(geometry.vertices, transform)
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        center = vertices.mean(axis=0)
        group, segment = classify(center, bounds, node)
        parts.append(
            {
                "node": node,
                "vertices": vertices,
                "faces": np.asarray(geometry.faces),
                "center": center,
                "group": group,
                "segment": segment,
            }
        )
    counts = {group: sum(part["group"] == group for part in parts) for group in COLORS}
    expected = {"trunk": 7, "head": 6, "tail": 6, "mandible_L": 4, "mandible_R": 4}
    expected.update({leg: 17 for leg in LEGS})
    if counts != expected:
        raise ValueError(f"Unexpected mk-1 grouping: {counts}")
    return parts


def part_by_node(parts: list[dict], node: str) -> dict:
    matches = [part for part in parts if part["node"] == node]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one GLB node named {node!r}")
    return matches[0]


def joint_landmarks(parts: list[dict]) -> dict:
    points = {
        "trunk": np.array([-1.45, 0.0, 0.0]),
        "head": part_by_node(parts, "Cube.002")["center"],
        "tail": part_by_node(parts, "Cube.009")["center"],
        "mandible_L": part_by_node(parts, "Plane.001")["center"],
        "mandible_R": part_by_node(parts, "Plane.039")["center"],
    }
    feet = {}
    for leg in LEGS:
        coxa = [part for part in parts if part["group"] == leg and part["segment"] == "coxa"]
        femur = [part for part in parts if part["group"] == leg and part["segment"] == "femur"]
        tibia = [part for part in parts if part["group"] == leg and part["segment"] == "tibia"]
        coxa_cubes = [part for part in coxa if part["node"].lower().startswith("cube")]
        femur_cubes = [part for part in femur if part["node"].lower().startswith("cube")]
        if len(coxa_cubes) != 2 or len(femur_cubes) != 1:
            raise ValueError(f"Could not identify the three joint markers of {leg}")
        points[f"{leg}_coxa"] = max(coxa_cubes, key=lambda part: part["center"][1])["center"]
        points[f"{leg}_femur"] = min(coxa_cubes, key=lambda part: part["center"][1])["center"]
        points[f"{leg}_tibia"] = femur_cubes[0]["center"]

        # Two 952-triangle Plane meshes are the terminal claw tips.
        claws = [part for part in tibia if part["node"].startswith("Plane") and len(part["faces"]) == 952]
        if len(claws) != 2:
            raise ValueError(f"Expected two terminal claws for {leg}; found {len(claws)}")
        sign = 1 if leg.startswith("L") else -1
        tips = []
        for claw in claws:
            vertices = claw["vertices"]
            radial = sign * vertices[:, 2]
            tip_vertices = vertices[radial >= radial.max() - 0.015]
            tips.append(tip_vertices.mean(axis=0))
        foot_center = np.mean(tips, axis=0)
        feet[leg] = {
            "claw_tips_world": [tip.round(6).tolist() for tip in tips],
            "contact_world": foot_center.round(6).tolist(),
            "contact_relative_to_knee": (foot_center - points[f"{leg}_tibia"]).round(6).tolist(),
        }
    return {"pivots": points, "feet": feet}


def concatenate(parts: list[dict], pivot: np.ndarray) -> trimesh.Trimesh:
    vertices = []
    faces = []
    offset = 0
    for part in parts:
        v = part["vertices"] - pivot
        f = part["faces"] + offset
        vertices.append(v)
        faces.append(f)
        offset += len(v)
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False
    )


def group_key(part: dict) -> str:
    return f"{part['group']}_{part['segment']}" if part["segment"] else part["group"]


def simplify_meshes(prepared: list[dict]) -> dict:
    script = Path(__file__).parent / "tools/mk1/simplify.mjs"
    if not (script.parent / "node_modules/meshoptimizer").exists():
        raise RuntimeError("Install the local reducer with: npm install --prefix veguinha_mini/tools/mk1")
    request = [
        {
            "key": item["key"],
            "vertices": item["mesh"].vertices.astype(np.float32).reshape(-1).tolist(),
            "indices": item["mesh"].faces.astype(np.uint32).reshape(-1).tolist(),
            "target": item["target"],
        }
        for item in prepared
    ]
    with tempfile.TemporaryDirectory(prefix="mk1-simplify-") as directory:
        source = Path(directory) / "input.json"
        result = Path(directory) / "output.json"
        source.write_text(json.dumps(request))
        subprocess.run(["node", str(script), str(source), str(result)], check=True)
        return json.loads(result.read_text())


def project(point: np.ndarray, parts: list[dict], axes: tuple[int, int, int], size: tuple[int, int]) -> tuple[float, float]:
    vertices = np.concatenate([part["vertices"] for part in parts])
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    u, v, _depth = axes
    width, height = size
    scale = min(
        (width - 160) / (bounds[1, u] - bounds[0, u]),
        (height - 170) / (bounds[1, v] - bounds[0, v]),
    )
    return (
        width / 2 + (point[u] - bounds[:, u].mean()) * scale,
        height / 2 - (point[v] - bounds[:, v].mean()) * scale,
    )


def mark_landmarks(
    image: Image.Image,
    parts: list[dict],
    landmarks: dict,
    axes: tuple[int, int, int],
    legs: tuple[str, ...] = LEGS,
) -> None:
    draw = ImageDraw.Draw(image)
    for leg in legs:
        hip = landmarks["pivots"][f"{leg}_coxa"]
        knee = landmarks["pivots"][f"{leg}_tibia"]
        foot = np.asarray(landmarks["feet"][leg]["contact_world"])
        hx, hy = project(hip, parts, axes, image.size)
        kx, ky = project(knee, parts, axes, image.size)
        fx, fy = project(foot, parts, axes, image.size)
        draw.rectangle((hx - 8, hy - 8, hx + 8, hy + 8), fill=(250, 234, 184), outline=(20, 24, 27), width=3)
        draw.ellipse((kx - 7, ky - 7, kx + 7, ky + 7), fill=(255, 255, 255), outline=(20, 24, 27), width=3)
        draw.line((fx - 10, fy - 10, fx + 10, fy + 10), fill=(15, 20, 24), width=3)
        draw.line((fx - 10, fy + 10, fx + 10, fy - 10), fill=(15, 20, 24), width=3)
    for jaw in ("mandible_L", "mandible_R"):
        x, y = project(landmarks["pivots"][jaw], parts, axes, image.size)
        draw.polygon(((x, y - 10), (x + 10, y), (x, y + 10), (x - 10, y)), fill=(255, 255, 255), outline=(20, 24, 27))


def make_preview(parts: list[dict], landmarks: dict, output: Path) -> None:
    visible = set(COLORS)
    top = render(parts, (0, 2, 1), visible, (1500, 1050), "mk-1 · montagem visual reduzida", color_fn=final_color)
    mark_landmarks(top, parts, landmarks, (0, 2, 1))
    top_draw = ImageDraw.Draw(top)
    top_draw.text((32, 65), "□ quadril    ○ joelho    × contato das garras    ◇ mandíbula", font=font(21), fill=(25, 32, 37))
    left = {"trunk", "head", "tail", "mandible_L", "mandible_R", "LF", "LM", "LH"}
    side_parts = [part for part in parts if part["group"] in left]
    side = render(parts, (0, 1, 2), left, (1500, 650), "Lateral · lado Z+", color_fn=final_color)
    mark_landmarks(side, side_parts, landmarks, (0, 1, 2), legs=("LF", "LM", "LH"))
    board = Image.new("RGB", (1500, 1700), (247, 248, 247))
    board.paste(top, (0, 0))
    board.paste(side, (0, 1050))
    board.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "storage/downloads/mk-1.glb")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/mk1_stage2")
    parser.add_argument("--target-triangles", type=int, default=36000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    parts = load_parts(args.input)
    landmarks = joint_landmarks(parts)
    original_total = sum(len(part["faces"]) for part in parts)
    # Reduce terminal claws first, so the dorsal plates can retain every
    # source triangle while the complete visual assembly stays below 40k.
    claws = [part for part in parts if part["segment"] == "tibia"
             and part["node"].startswith("Plane") and len(part["faces"]) == 952]
    claw_requests = []
    for i, part in enumerate(claws):
        mesh = trimesh.Trimesh(vertices=part["vertices"], faces=part["faces"], process=False)
        mesh.merge_vertices()
        claw_requests.append({"key": f"claw_{i}", "mesh": mesh, "target": 600})
    claw_results = simplify_meshes(claw_requests)
    for i, part in enumerate(claws):
        part["vertices"] = claw_requests[i]["mesh"].vertices.copy()
        part["faces"] = np.asarray(claw_results[f"claw_{i}"]["indices"], dtype=np.int64).reshape(-1, 3)
    grouped = defaultdict(list)
    for part in parts:
        grouped[group_key(part)].append(part)
    prepared = []
    for key, members in grouped.items():
        group, segment = members[0]["group"], members[0]["segment"]
        pivot = landmarks["pivots"][key]
        mesh = concatenate(members, pivot)
        mesh.merge_vertices()
        before = len(mesh.faces)
        target = before if group == "trunk" else max(40, round(before * args.target_triangles / original_total))
        prepared.append(
            {"key": key, "group": group, "segment": segment, "pivot": pivot,
             "mesh": mesh, "before": before, "target": target, "members": members}
        )

    simplified = simplify_meshes(prepared)
    scene = trimesh.Scene()
    preview_parts = []
    output_groups = []
    for item in prepared:
        key, group, segment = item["key"], item["group"], item["segment"]
        pivot, before, members = item["pivot"], item["before"], item["members"]
        result = simplified[key]
        reduced_faces = np.asarray(result["indices"], dtype=np.int64).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=item["mesh"].vertices.copy(), faces=reduced_faces, process=False)
        mesh.remove_unreferenced_vertices()
        color = np.asarray(final_color(group, segment), dtype=float)
        rgba = np.r_[color.astype(np.uint8), 255]
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=key, baseColorFactor=rgba, metallicFactor=0.65, roughnessFactor=0.48
            )
        )
        transform = np.eye(4)
        transform[:3, 3] = pivot
        scene.add_geometry(mesh, node_name=key, geom_name=key, transform=transform)
        preview_parts.append(
            {"vertices": mesh.vertices + pivot, "faces": mesh.faces, "group": group, "segment": segment}
        )
        output_groups.append(
            {
                "group": key,
                "source_pieces": len(members),
                "triangles_before": before,
                "triangles_after": len(mesh.faces),
                "relative_simplification_error": float(result["error"]),
                "pivot_world": np.asarray(pivot).round(6).tolist(),
                "color_rgba": rgba.tolist(),
            }
        )

    glb_path = args.output / "mk-1-visual.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))
    make_preview(preview_parts, landmarks, args.output / "resultado.png")
    report = {
        "source": str(args.input),
        "visual_glb": str(glb_path),
        "triangles_before": original_total,
        "triangles_after": sum(group["triangles_after"] for group in output_groups),
        "claw_triangles_before": 952,
        "claw_triangles_after_prefilter": [len(part["faces"]) for part in claws],
        "groups": output_groups,
        "jaw_pivots_world": {key: value.round(6).tolist() for key, value in landmarks["pivots"].items() if key.startswith("mandible")},
        "leg_pivots_world": {key: value.round(6).tolist() for key, value in landmarks["pivots"].items() if key[:2] in LEGS},
        "feet": landmarks["feet"],
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Reload the exported file: the hierarchy, pivots, colors and triangle budget
    # must survive GLB serialization, not only exist in memory.
    exported = trimesh.load(glb_path, force="scene")
    if len(exported.graph.nodes_geometry) != len(grouped):
        raise ValueError("GLB export lost a visual group")
    faces = sum(len(exported.geometry[exported.graph[node][1]].faces) for node in exported.graph.nodes_geometry)
    if not 30000 <= faces <= 40000:
        raise ValueError(f"GLB has {faces} triangles, outside the requested budget")
    print(json.dumps({"groups": len(grouped), "triangles_before": original_total, "triangles_after": faces, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
