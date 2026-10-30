"""Independent Viser renderer of the actual MuJoCo state published by Simulation."""

import threading
import time
import copy

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

from .body import LEG_NAMES
from .mujoco_body import MJCF_PATH


class MjviserBridge:
    """Viewer owns a read-only MjData; only the simulation thread calls mj_step."""

    def __init__(self, simulation, port=8002, fps=3):
        self.simulation = simulation
        self.port = port
        self.target_fps = fps
        self.active = True
        self.force_render = False
        self.stopped = threading.Event()
        self.frames = 0
        self.render_seconds = 0.0
        self.last_render_wall = 0.0
        source=getattr(getattr(simulation,'body',None),'model',None)
        self.model = copy.copy(source) if source is not None else mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.toe_ids = [self.model.site(f"{leg}_toe").id for leg in LEG_NAMES]
        self.server = viser.ViserServer(host="127.0.0.1", port=port,
                                       label="Veguinha mini 3D", verbose=False)
        @self.server.on_client_connect
        def _camera(client):
            time.sleep(.8)
            try:
                client.camera.up_direction = (0.0, 0.0, 1.0)
                latest=getattr(self.simulation,'latest',None)
                position=np.array(latest()['body']['mj_state']['qpos'][:3] if latest else [-1.2,0,.18])
                client.camera.position = tuple(position+np.array([-1.2,-1.4,1.]))
                client.camera.look_at = tuple(position)
            except Exception:
                pass
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        # MuJoCo geometry and our odor/food overlays share world coordinates.
        self.scene.camera_tracking_enabled = False
        self.floor_grid=self.server.scene.add_grid('/overlay/floor_grid',width=6,height=4,
            plane='xy',cell_size=.25,section_size=1.,cell_color=(60,80,90),
            section_color=(60,80,90),cell_thickness=1.,section_thickness=1.,
            position=(0,0,.003),fade_from='origin',fade_distance=100.)
        self.contacts = [self.server.scene.add_icosphere(
            f"/overlay/contact/{leg}", radius=.025, color=(116, 145, 153),
            position=(0, 0, .03)) for leg in LEG_NAMES]
        self.food = self.server.scene.add_icosphere(
            "/overlay/food", radius=.065, color=(243, 188, 104),
            position=(0, 0, .07), subdivisions=2)
        xs = np.linspace(-2.9, 2.9, 31)
        ys = np.linspace(-1.9, 1.9, 21)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        self.odor_points = np.column_stack((grid, np.full(len(grid), .012))).astype(np.float32)
        self.odor = self.server.scene.add_point_cloud(
            "/overlay/odor", self.odor_points,
            np.zeros((len(grid), 3), dtype=np.uint8), point_size=.035,
            point_shape="circle")
        self.path = []
        self.path_line = self.server.scene.add_line_segments(
            "/overlay/trajectory", np.zeros((0, 2, 3), dtype=np.float32),
            colors=(101, 221, 195), thickness=.012)
        self.last_epoch = None
        self.last_food = object()
        sx,sy=np.meshgrid(np.linspace(-2.85,-1.15,45),np.linspace(.45,1.75,35))
        self.shade_points=np.column_stack([sx.ravel(),sy.ravel(),np.full(sx.size,.006)]).astype(np.float32)
        self.shade=self.server.scene.add_point_cloud('/overlay/shade',self.shade_points,
            np.tile(np.array([12,18,24],np.uint8),(len(self.shade_points),1)),point_size=.05,point_shape='square')
        self.shade.visible=False
        rng=np.random.default_rng(8116)
        self.air_points=rng.uniform([-3,-2,.03],[3,2,.8],(96,3)).astype(np.float32)
        self.air=self.server.scene.add_point_cloud('/overlay/air_dust',self.air_points,
            np.tile(np.array([180,170,145],np.uint8),(96,1)),point_size=.006,point_shape='circle')
        self.air.visible=False
        self.thread = threading.Thread(target=self._loop, name="veguinha-mjviser", daemon=True)

    @staticmethod
    def _xy(point):
        return np.array([(point[0] - 300.0) / 100.0,
                         (200.0 - point[1]) / 100.0], dtype=float)

    def render(self, frame):
        body = frame["body"]
        state = body["mj_state"]
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)
        for index, marker in enumerate(self.contacts):
            marker.position = self.data.site_xpos[self.toe_ids[index]].copy()
            marker.color = ((101, 221, 195) if body["contacts"][index]
                            else (116, 145, 153))
        world = frame["world"]
        self.shade.visible=world.get('shade') is not None
        ecology=body.get('ecology');self.air.visible=ecology is not None
        if ecology is not None:
            points=self.air_points.copy();points[:,0]=(points[:,0]+.025*self.data.time+3)%6-3
            points[:,2]=.03+(points[:,2]-.008*self.data.time)%.77;self.air.points=points
        stock=float(np.clip(world.get('food_energy',1.),0,1))
        self.food.color=tuple(int(v*(.4+.6*stock)) for v in (243,188,104))
        odor = world["odor"]
        if odor != self.last_food:
            self.last_food = odor
            self.food.visible = odor is not None
            self.odor.visible = odor is not None
            if odor is not None:
                self.food.position = [*self._xy(odor), .065]
                delta = self.odor_points[:, :2] - self._xy(odor)
                intensity = np.exp(-np.sum(delta * delta, axis=1) /
                                   (2 * (world["odor_sigma"] / 100) ** 2))
                self.odor.colors = np.column_stack((
                    28 + 185 * intensity, 36 + 113 * intensity,
                    42 + 38 * intensity)).astype(np.uint8)
        if frame["epoch"] != self.last_epoch:
            self.path.clear()
            self.last_epoch = frame["epoch"]
        position = np.array([*self.data.qpos[:2], .035], dtype=np.float32)
        if not self.path or np.linalg.norm(position - self.path[-1]) > .015:
            self.path.append(position)
            self.path = self.path[-300:]
            if len(self.path) > 1:
                self.path_line.points = np.stack((self.path[:-1], self.path[1:]), axis=1)

    def _loop(self):
        last_seq = -1
        while not self.stopped.is_set():
            start = time.perf_counter()
            frame = self.simulation.latest()
            clients = bool(self.server.get_clients())
            due = clients or self.force_render or start - self.last_render_wall >= 1
            if (self.active and due and frame.get("speed") != "turbo"
                    and frame["seq"] != last_seq):
                try:
                    self.render(frame)
                    last_seq = frame["seq"]
                    self.frames += 1
                    self.render_seconds += time.perf_counter() - start
                    self.last_render_wall = time.perf_counter()
                except Exception as exc:
                    print(f"mjviser: erro de renderização: {exc}", flush=True)
                    self.active = False
            interval = max(1 / self.target_fps if clients or self.force_render else 1,
                           2 * (time.perf_counter() - start))
            self.stopped.wait(interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.server.stop()
