"""Build the articulated mk-1 MJCF; the two-joint control is never overwritten."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

ROOT = Path(__file__).parent
MK1_PATH = ROOT / "assets/hexapod_mk1.xml"
JOINTS = ("coxa", "femur", "tibia")


def vec(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def build():
    manifest = json.loads((ROOT / "results/mk1_stage2/manifest.json").read_text())
    plan = json.loads((ROOT / "results/mk1_stage3_plan/plan.json").read_text())
    scene = trimesh.load(ROOT / "results/mk1_stage2/mk-1-visual.glb", force="scene")
    scale = plan["scale_m_per_glb_unit"]
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    reference = np.array([-1.45, 0, 0])

    def point(p):
        return scale * rotation @ (np.asarray(p) - reference)

    pivots = {g["group"]: np.asarray(g["pivot_world"]) for g in manifest["groups"]}
    assets = ROOT / "assets/mk1"
    assets.mkdir(exist_ok=True)
    # Copy arena/options only. The source control model stays byte-identical.
    xml = ET.parse(ROOT / "assets/hexapod.xml").getroot()
    xml.set("model", "veguinha_mk1_three_active_joints")
    asset = xml.find("asset")
    asset.clear()
    world = xml.find("worldbody")
    world.remove(world.find("body"))
    xml.find("actuator").clear()
    motors = xml.find("actuator")
    thorax = ET.SubElement(world, "body", name="thorax", pos="-1.2 0 .1804")
    ET.SubElement(thorax, "freejoint", name="base")

    def visual(parent, key, origin):
        transform, geometry = scene.graph[key]
        mesh = scene.geometry[geometry].copy()
        vertices = trimesh.transform_points(mesh.vertices, transform)
        mesh.vertices = (scale * (vertices - origin)) @ rotation.T
        (assets / f"{key}.obj").write_text(mesh.export(file_type="obj"))
        ET.SubElement(asset, "mesh", name=key, file=f"mk1/{key}.obj")
        color = scene.geometry[geometry].visual.material.baseColorFactor / 255
        ET.SubElement(thorax if parent is None else parent, "geom",
                      name=f"visual_{key}", type="mesh", mesh=key, mass="0",
                      contype="0", conaffinity="0", group="1", rgba=vec(color))

    for key in ("trunk", "head", "tail", "mandible_L", "mandible_R"):
        visual(thorax, key, reference)
    for name, center, size, mass in (
        ("shell", [-1.342739, .255957, .013508], [.205, .075, .052], .196),
        ("head", [.072938, .187809, .000872], [.065, .065, .045], .016),
        ("tail", [-2.643553, -.291997, .004826], [.062, .034, .025], .008),
    ):
        ET.SubElement(thorax, "geom", name=name, type="ellipsoid",
                      pos=vec(point(center)), size=vec(size), mass=str(mass), rgba=".2 .3 .3 0")
    for leg, leg_plan in plan["legs"].items():
        leg_lower = leg.lower()
        parent = thorax
        last_point = np.zeros(3)
        for index, joint in enumerate(JOINTS):
            key = f"{leg}_{joint}"
            origin = point(pivots[key])
            link = ET.SubElement(parent, "body", name=f"{leg_lower}_{joint}",
                                 pos=vec(origin - last_point))
            ET.SubElement(link, "joint", name=f"{leg_lower}_{joint}", type="hinge",
                          axis=vec(leg_plan["joint_axes_mujoco"][joint]),
                          range="-.8 .8", stiffness=str((.2, .8, .8)[index]),
                          damping=str((.025, .045, .045)[index]))
            if index < 2:
                end = point(pivots[f"{leg}_{JOINTS[index + 1]}"]) - origin
                mass = .005
            else:
                end = point(leg_plan["foot_world_glb"]) - origin
                # Source claw heights differ by ~1 mm. Collision centers share
                # one neutral plane; the visual mesh keeps the source geometry.
                end[2] = .024 - .1804 - origin[2]
                mass = .004
            ET.SubElement(link, "geom", name=f"{leg_lower}_{joint}_shaft",
                          type="capsule", fromto=vec([0, 0, 0, *end]),
                          size=str((.009, .010, .008)[index]), mass=str(mass), rgba=".3 .4 .4 0")
            visual(link, key, pivots[key])
            if index == 2:
                ET.SubElement(link, "geom", name=f"{leg_lower}_foot", type="sphere",
                              pos=vec(end), size=".024", mass=".002",
                              friction="1.4 .02 .002", rgba=".7 .6 .3 0")
                ET.SubElement(link, "site", name=f"{leg_lower}_toe", pos=vec(end), size=".005")
            ET.SubElement(motors, "motor", name=f"{leg_lower}_{joint}_motor",
                          # The femur must unload the foot against its .8 Nm/rad
                          # passive spring; the old hip gear pinned it to ground.
                          joint=f"{leg_lower}_{joint}", gear=str((.08, .2, .2)[index]),
                          ctrllimited="true", ctrlrange="-1 1")
            parent, last_point = link, origin
    ET.indent(xml)
    ET.ElementTree(xml).write(MK1_PATH, encoding="unicode")
    # Neutral passive spring preload balances gravity and six equal normal
    # forces. This is static mechanical calibration, never a motor command.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MK1_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    toes = [model.site(f"{leg.lower()}_toe").id for leg in plan["legs"]]
    shift = (data.site_xpos[toes].mean(axis=0) - data.subtree_com[1])[:2]
    shell = thorax.find("geom[@name='shell']")
    shell_pos = np.fromstring(shell.get("pos"), sep=" ")
    shell_pos[:2] += shift * .316 / .196
    shell.set("pos", vec(shell_pos))
    ET.ElementTree(xml).write(MK1_PATH, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(MK1_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    supported = np.zeros(model.nv)
    for toe in toes:
        body_id = model.site_bodyid[toe]
        mujoco.mj_applyFT(model, data, np.array([0., 0., .316 * 9.81 / 6]),
                          np.zeros(3), data.site_xpos[toe], body_id, supported)
    required = data.qfrc_bias - supported
    for joint in xml.findall(".//joint[@type='hinge']"):
        joint_id = model.joint(joint.get("name")).id
        joint.set("springref", str(required[model.jnt_dofadr[joint_id]] /
                                    model.jnt_stiffness[joint_id]))
    ET.indent(xml)
    ET.ElementTree(xml).write(MK1_PATH, encoding="unicode")
    return MK1_PATH


if __name__ == "__main__":
    print(build())
