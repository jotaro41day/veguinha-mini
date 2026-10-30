"""Stage 1 visual audit of the mk-1.glb parts; no physical model is changed.

Run: python -m veguinha_mini.mk1_grouping --input ~/storage/downloads/mk-1.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


# GLB coordinates: X points to the head, Y is vertical, Z is lateral.
# Side names are provisional until the physical coordinate frame is fitted.
LEG_ANCHORS = {"F": -0.63, "M": -1.39, "H": -2.10}
# Midpoints between the radial clusters of the first and second bends.
RADIAL_CUTS = {"F": (0.64, 1.05), "M": (0.70, 1.21), "H": (0.46, 0.86)}
COLORS = {
    "trunk": (89, 112, 122),
    "head": (244, 211, 94),
    "tail": (120, 97, 165),
    "mandible_L": (242, 140, 40),
    "mandible_R": (232, 110, 152),
    "LF": (199, 31, 55),
    "LM": (0, 166, 200),
    "LH": (138, 109, 59),
    "RF": (40, 105, 211),
    "RM": (44, 158, 80),
    "RH": (157, 68, 192),
}
SEGMENT_FACTOR = {"coxa": 0.72, "femur": 1.0, "tibia": 1.24}
SEGMENT_COLORS = {"coxa": (65, 138, 204), "femur": (193, 92, 167), "tibia": (58, 164, 104)}


def classify(center: np.ndarray, bounds: np.ndarray, name: str) -> tuple[str, str | None]:
    x, y, z = center
    radius = abs(z)
    # Four paired mouth parts per side sit below and ahead of the head.
    # Eyes and antennae are higher and remain attached to the head.
    if x > -0.12 and y < -0.10 and 0.06 < radius < 0.66:
        return ("mandible_L" if z > 0 else "mandible_R"), None
    # Central pieces at both ends are reserved before nearest-leg assignment.
    # This includes mandibles and antennae whose centroids are off-axis.
    if name.strip() == "cabeça" or (x > -0.55 and radius < 0.56):
        return "head", None
    if x < -2.25 and radius < 0.43:
        return "tail", None
    if radius < 0.22 or (bounds[0, 2] < -0.12 and bounds[1, 2] > 0.12):
        return "trunk", None

    station = min(LEG_ANCHORS, key=lambda key: abs(x - LEG_ANCHORS[key]))
    side = "L" if z > 0 else "R"
    inner, outer = RADIAL_CUTS[station]
    segment = "coxa" if radius < inner else "femur" if radius < outer else "tibia"
    return side + station, segment


def font(size: int) -> ImageFont.ImageFont:
    path = "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def render(
    parts: list[dict],
    axes: tuple[int, int, int],
    included: set[str],
    size: tuple[int, int],
    title: str,
    segment_palette: bool = False,
    depth_reverse: bool = False,
    color_fn=None,
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (247, 248, 247))
    draw = ImageDraw.Draw(image)
    selected = [part for part in parts if part["group"] in included]
    all_vertices = np.concatenate([part["vertices"] for part in selected])
    bounds = np.array([all_vertices.min(axis=0), all_vertices.max(axis=0)])
    u_axis, v_axis, depth_axis = axes
    margin_x, margin_y = 80, 105
    u_min, u_max = bounds[:, u_axis]
    v_min, v_max = bounds[:, v_axis]
    scale = min(
        (width - 2 * margin_x) / (u_max - u_min),
        (height - margin_y - 65) / (v_max - v_min),
    )
    u_center = (u_min + u_max) / 2
    v_center = (v_min + v_max) / 2
    faces = []
    for part in selected:
        vertices = part["vertices"]
        tris = vertices[part["faces"]]
        color = np.asarray(
            color_fn(part["group"], part["segment"]) if color_fn is not None else
            SEGMENT_COLORS[part["segment"]]
            if segment_palette and part["segment"] else COLORS[part["group"]], dtype=float,
        )
        if part["segment"] and not segment_palette and color_fn is None:
            color = np.clip(color * SEGMENT_FACTOR[part["segment"]], 0, 255)
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        shade = 0.75 + 0.23 * np.abs(normal[:, depth_axis])
        projected_u = width / 2 + (tris[:, :, u_axis] - u_center) * scale
        projected_v = height / 2 - (tris[:, :, v_axis] - v_center) * scale
        for i in range(len(tris)):
            polygon = [
                (float(projected_u[i, j]), float(projected_v[i, j])) for j in range(3)
            ]
            rgb = tuple(np.clip(color * shade[i], 0, 255).astype(np.uint8))
            faces.append((float(tris[i, :, depth_axis].mean()), polygon, rgb))

    # Far faces first, for an orthographic, static inspection image.
    for _depth, polygon, color in sorted(
        faces, key=lambda entry: entry[0], reverse=depth_reverse
    ):
        draw.polygon(polygon, fill=color)
    draw.text((30, 20), title, font=font(33), fill=(30, 39, 45))
    draw.text((30, height - 47), "X+ → cabeça", font=font(20), fill=(67, 76, 81))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "storage/downloads/mk-1.glb")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/mk1_stage1"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(args.input, force="scene")
    parts = []
    manifest = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        vertices = trimesh.transform_points(geometry.vertices, transform)
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        center = vertices.mean(axis=0)
        group, segment = classify(center, bounds, node)
        parts.append({"vertices": vertices, "faces": geometry.faces, "group": group, "segment": segment})
        manifest.append(
            {
                "node": node,
                "geometry": geometry_name,
                "group": group,
                "segment": segment,
                "centroid": center.round(5).tolist(),
                "bounds": bounds.round(5).tolist(),
                "triangles": int(len(geometry.faces)),
            }
        )

    groups = {group: sum(part["group"] == group for part in manifest) for group in COLORS}
    triangles = {group: sum(part["triangles"] for part in manifest if part["group"] == group) for group in COLORS}
    report = {
        "input": str(args.input),
        "axes": {"front": "+X", "up": "+Y", "left_provisional": "+Z"},
        "leg_stations_x": LEG_ANCHORS,
        "segment_radial_cuts": RADIAL_CUTS,
        "parts": len(manifest),
        "triangles": int(sum(part["triangles"] for part in manifest)),
        "group_piece_counts": groups,
        "group_triangle_counts": triangles,
        "mapping": manifest,
    }
    (args.output / "mapping.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    all_groups = set(COLORS)
    top = render(parts, (0, 2, 1), all_groups, (1500, 1050), "mk-1 · vista superior · grupos e segmentos")
    top_draw = ImageDraw.Draw(top)
    for i, group in enumerate(COLORS):
        x = 22
        y = 75 + i * 30
        top_draw.rectangle((x, y, x + 24, y + 22), fill=COLORS[group])
        top_draw.text((x + 34, y), group, font=font(20), fill=(35, 44, 50))
    top.save(args.output / "vista_superior.png")

    side = Image.new("RGB", (1500, 1300), (247, 248, 247))
    for index, sign in enumerate(("L", "R")):
        visible = {"trunk", "head", "tail", "mandible_L", "mandible_R"} | {
            group for group in COLORS if group.startswith(sign)
        }
        panel = render(
            parts,
            (0, 1, 2),
            visible,
            (1500, 650),
            f"Vista lateral · lado {sign} (Z{'+' if sign == 'L' else '−'})",
            depth_reverse=sign == "R",
        )
        side.paste(panel, (0, index * 650))
    side.save(args.output / "vista_lateral.png")
    segments = render(
        parts,
        (0, 2, 1),
        all_groups,
        (1500, 1050),
        "mk-1 · três segmentos por perna",
        segment_palette=True,
    )
    segment_draw = ImageDraw.Draw(segments)
    for index, (segment, color) in enumerate(SEGMENT_COLORS.items()):
        x = (40, 340, 950)[index]
        segment_draw.rectangle((x, 75, x + 28, 98), fill=color)
        segment_draw.text((x + 38, 74), segment, font=font(22), fill=(35, 44, 50))
    segments.save(args.output / "vista_segmentos.png")
    print(json.dumps({"groups": groups, "triangles": triangles, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
