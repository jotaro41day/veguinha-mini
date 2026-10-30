"""Render saved MuJoCo poses; never advances the simulation or invents a gait."""
from pathlib import Path
import numpy as np
import mujoco
import trimesh
from PIL import Image, ImageDraw
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.tools.render_model_comparison import render_panel, font

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mk1_walk_audit_v1'


def panel(gear, label):
    body = MujocoBody(variant='mk1')
    with np.load(OUT/f'42_tripod_f{gear:g}_p0.38.npz') as trace:
        body.data.qpos[:] = trace['qpos'][750]
        loaded = trace['load'][750] > .02
    mujoco.mj_forward(body.model, body.data)
    angle = np.deg2rad(52)
    view = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
    meshes = []
    for g in range(body.model.ngeom):
        if body.model.geom_group[g] != 1:
            continue
        mesh = int(body.model.geom_dataid[g])
        start, count = body.model.mesh_vertadr[mesh], body.model.mesh_vertnum[mesh]
        face, nface = body.model.mesh_faceadr[mesh], body.model.mesh_facenum[mesh]
        vertices = body.model.mesh_vert[start:start+count].astype(float)
        vertices = vertices @ body.data.geom_xmat[g].reshape(3, 3).T + body.data.geom_xpos[g]
        color = tuple(np.rint(255*body.model.geom_rgba[g, :3]).astype(int))
        meshes.append((vertices @ view.T, body.model.mesh_face[face:face+nface], color))
    assert len(meshes) == 23
    for i, site in enumerate(body.toe_ids):
        marker = trimesh.creation.icosphere(subdivisions=1, radius=.011)
        vertices = (marker.vertices+body.data.site_xpos[site]) @ view.T
        meshes.append((vertices, marker.faces, (45, 160, 115) if loaded[i] else (225, 140, 35)))
    return render_panel(meshes, label, f'Fêmur: {gear:.2f} N·m',
                        'Estado físico salvo · t = 15,02 s', (55, 105, 105))


if __name__ == '__main__':
    image = Image.new('RGB', (1840, 1010), (234, 239, 236))
    draw = ImageDraw.Draw(image)
    draw.text((40, 22), 'VEGUINHA · CORREÇÃO MECÂNICA', font=font(31, True), fill=(33, 62, 64))
    draw.text((40, 65), 'Sonda senoidal sem rede e sem aprendizado · mesma entrada motora · seed 42',
              font=font(21), fill=(68, 91, 94))
    image.paste(panel(.08, 'ANTES'), (20, 118))
    image.paste(panel(.2, 'DEPOIS'), (930, 118))
    draw.text((40, 980), 'Pontos verdes: pé carregado. Laranja: pé descarregado. Poses extraídas dos NPZ do ensaio.',
              font=font(18), fill=(68, 91, 94))
    image.save(OUT/'correcao_mecanica.png', optimize=True)
    print(OUT/'correcao_mecanica.png')
