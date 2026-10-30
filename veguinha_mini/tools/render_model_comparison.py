"""Render the original pre-Ladybird body beside the approved mk-1 visual GLB.

This is a geometry-based orthographic illustration, not a neural/physics run.
"""

from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "veguinha_mini"
OUT = PKG / "results" / "mk1_stage2" / "antes_vs_depois.png"
FONT = Path("/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf")
FONT_BOLD = FONT.with_name("DejaVuSans-Bold.ttf")
SIZE = (1900, 1060)


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size)


def capsule_mesh(pos, matrix, radius, half_length):
    mesh = trimesh.creation.capsule(height=2 * half_length, radius=radius, count=[8, 8])
    transform = np.eye(4)
    transform[:3, :3] = matrix
    transform[:3, 3] = pos
    mesh.apply_transform(transform)
    return mesh


def old_meshes():
    model = mujoco.MjModel.from_xml_path(str(PKG / "assets" / "hexapod.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        # The original pre-skin geoms remain in the MJCF. Their alpha was set
        # to zero only when the Ladybird skin was added, so ignore alpha here.
        if model.geom_bodyid[geom_id] == 0 or model.geom_group[geom_id] != 0:
            continue
        kind = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(height=2 * float(size[1]),
                                            radius=float(size[0]), count=[10, 10])
        elif (kind == mujoco.mjtGeom.mjGEOM_SPHERE or
              kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            mesh = trimesh.creation.icosphere(subdivisions=3)
            scale = np.repeat(size[0], 3) if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size
            mesh.vertices *= scale
        else:
            continue
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[geom_id]
        mesh.apply_transform(transform)
        color = tuple(int(round(255 * x)) for x in model.geom_rgba[geom_id, :3])
        if name == "shell":
            color = (57, 146, 142)
        elif name == "head":
            color = (91, 185, 157)
        result.append((np.asarray(mesh.vertices), np.asarray(mesh.faces), color))
    return result


def new_meshes():
    scene = trimesh.load(PKG / "results" / "mk1_stage2" / "mk-1-visual.glb", force="scene")
    result = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry]
        source = trimesh.transform_points(mesh.vertices, transform)
        # GLB X is forward, Y up and Z left; MuJoCo uses X forward and Z up.
        vertices = np.column_stack((
            -1.2 + .17 * (source[:, 0] + 1.45),
            -.17 * source[:, 2],
            .024 + .17 * (source[:, 1] + .92),
        ))
        color = tuple(int(x) for x in mesh.visual.material.baseColorFactor[:3])
        result.append((vertices, np.asarray(mesh.faces), color))
    return result


def render_panel(meshes, label, title, caption, accent):
    width, height = 890, 860
    image = Image.new("RGB", (width, height), (248, 249, 247))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((17, 16, width - 17, height - 16), radius=24,
                           fill=(254, 254, 252), outline=(221, 226, 222), width=2)
    draw.rounded_rectangle((46, 44, 192, 86), radius=12, fill=accent)
    draw.text((66, 53), label, font=font(24, True), fill=(255, 255, 255))
    draw.text((48, 110), title, font=font(33, True), fill=(31, 46, 51))
    draw.text((48, 158), caption, font=font(19), fill=(89, 101, 104))

    points = np.concatenate([mesh[0][:, :2] for mesh in meshes])
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    span = bounds[1] - bounds[0]
    scale = min(760 / span[0], 610 / span[1])
    center = bounds.mean(axis=0)
    x0, y0 = width / 2, 525

    def project(v):
        return np.column_stack((x0 + (v[:, 0] - center[0]) * scale,
                                y0 - (v[:, 1] - center[1]) * scale))

    # The shadow is a graphic cue; all body/limb geometry comes from the assets.
    draw.ellipse((145, 320, 765, 745), fill=(239, 242, 239))
    faces = []
    light = np.array([-.38, -.24, .893])
    light /= np.linalg.norm(light)
    for vertices, triangles, base in meshes:
        xyz = vertices[triangles]
        screen = project(vertices)[triangles]
        normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        shades = .70 + .28 * np.abs(normals @ light)
        for screen_face, world_face, shade in zip(screen, xyz, shades):
            color = tuple(int(np.clip(channel * shade, 0, 255)) for channel in base)
            polygon = [tuple(map(float, corner)) for corner in screen_face]
            faces.append((float(world_face[:, 2].mean()), polygon, color))
    for _, polygon, color in sorted(faces, key=lambda face: face[0]):
        draw.polygon(polygon, fill=color)
    draw.text((48, 806), "Cabeça  →", font=font(20, True), fill=(88, 99, 100))
    return image


def main():
    old = render_panel(old_meshes(), "ANTES", "Corpo original", "Antes da joaninha · 2 juntas por perna", (65, 138, 139))
    new = render_panel(new_meshes(), "DEPOIS", "mk-1 metálico", "Visual aprovado · física de 3 juntas pendente", (50, 104, 102))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old.save(OUT.with_name("modelo_original.png"), optimize=True)
    new.save(OUT.with_name("modelo_mk1.png"), optimize=True)
    canvas = Image.new("RGB", SIZE, (234, 239, 236))
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 24), "VEGUINHA MINI", font=font(31, True), fill=(33, 62, 64))
    draw.text((60, 68), "Comparação visual dos modelos · vista superior · escala individual por quadro",
              font=font(20), fill=(78, 96, 98))
    canvas.paste(old, (40, 148))
    canvas.paste(new, (970, 148))
    canvas.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
