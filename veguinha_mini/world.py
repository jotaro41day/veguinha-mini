"""Fixed world coordinates, odor and a finite nutrient resource."""

from dataclasses import dataclass,replace

import numpy as np


@dataclass(frozen=True)
class World:
    width: float = 600.0
    height: float = 400.0
    odor_x: float = 450.0
    odor_y: float = 190.0
    odor_sigma: float = 85.0
    food_enabled: bool = True
    food_energy: float = 1.0
    food_radius_m: float = .065
    shade_enabled: bool = False

    def illumination_at(self,points):
        points=np.asarray(points,float)
        if not self.shade_enabled:return np.ones(points.shape[:-1])
        relative=(points[...,:2]-np.array([-2.,1.1]))/np.array([.85,.65])
        return 1-.93*np.exp(-np.sum(relative**4,axis=-1))

    @property
    def food_position_m(self):
        return np.array([(self.odor_x-300)/100,(200-self.odor_y)/100,self.food_radius_m])

    def consume(self,amount):
        taken=float(np.clip(amount,0,max(0.,self.food_energy))) if self.food_enabled else 0.
        return replace(self,food_energy=max(0.,self.food_energy-taken)),taken

    def odor_at(self, points):
        points = np.asarray(points, dtype=float)
        if not self.food_enabled or self.food_energy<=0:
            return np.zeros(points.shape[:-1], dtype=float)
        displacement = points - (self.odor_x, self.odor_y)
        return np.exp(-np.sum(displacement * displacement, axis=-1) /
                      (2 * self.odor_sigma ** 2))

    def snapshot(self):
        return {"width": self.width, "height": self.height,
                "odor": [self.odor_x, self.odor_y] if self.food_enabled and self.food_energy>0 else None,
                "odor_sigma": self.odor_sigma,"food_energy":self.food_energy,
                "food_radius_m":self.food_radius_m,
                "shade":dict(center_m=[-2.,1.1],radius_m=[.85,.65]) if self.shade_enabled else None}
