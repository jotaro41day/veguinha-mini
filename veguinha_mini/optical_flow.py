"""Local image motion from two successive retinas; no pose or scene labels."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class OpticalFlow:
    rows, cols = 3, 8

    def __init__(self):
        self.previous = None
        self.result = self.empty()

    def empty(self):
        return dict(angular_rad_s=np.zeros((2, self.rows, self.cols, 2)).tolist(),
                    confidence=np.zeros((2, self.rows, self.cols)).tolist(),
                    valid_fraction=0., units='rad/s (image axes)')

    def update(self, gray, dt, focal_px):
        gray = np.asarray(gray, float)
        # Area averaging bounds cost, while keeping the camera's full RGB image.
        stride = max(1, gray.shape[2] // 160)
        h, w = gray.shape[1] // stride, gray.shape[2] // stride
        image = gray[:, :h*stride, :w*stride].reshape(2, h, stride, w, stride).mean((2, 4))
        old, self.previous = self.previous, image.copy()
        self.result = self.empty()
        if old is None or old.shape != image.shape or dt is None or dt <= 0:
            return self.result
        velocities = np.zeros((2, self.rows, self.cols, 2))
        confidence = np.zeros((2, self.rows, self.cols))
        radius=min(5,max(2,w//32),max(1,(h-1)//2))
        search=min(7,max(3,w//20))
        size = 2*radius+1
        for eye in range(2):
            for row, y in enumerate(np.linspace(radius, h-radius-1, self.rows).astype(int)):
                for col, x in enumerate(np.linspace(radius, w-radius-1, self.cols).astype(int)):
                    patch = old[eye, y-radius:y+radius+1, x-radius:x+radius+1]
                    patch = patch-patch.mean()
                    norm = np.linalg.norm(patch)
                    if norm*norm/patch.size < 1e-4:
                        continue
                    top, bottom = max(radius, y-search), min(h-radius-1, y+search)
                    left, right = max(radius, x-search), min(w-radius-1, x+search)
                    candidates = sliding_window_view(image[eye, top-radius:bottom+radius+1,
                                                           left-radius:right+radius+1], (size, size))
                    centered = candidates-candidates.mean((-2, -1), keepdims=True)
                    norms = np.linalg.norm(centered, axis=(-2, -1))
                    scores = np.sum(centered*patch, axis=(-2, -1))/np.maximum(norms*norm, 1e-12)
                    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
                    peak = scores[iy, ix]
                    rivals = scores.copy()
                    rivals[max(0, iy-1):iy+2, max(0, ix-1):ix+2] = -1
                    margin = peak-rivals.max()
                    if peak < .65 or margin < .05:
                        continue
                    offset = np.array([left+ix-x, top+iy-y], float)
                    # Parabolic interpolation around the best integer match.
                    for axis, index in enumerate((ix, iy)):
                        line = scores[iy] if axis == 0 else scores[:, ix]
                        if 0 < index < len(line)-1:
                            a, b, c = line[index-1:index+2]
                            offset[axis] += np.clip(.5*(a-c)/min(a-2*b+c, -1e-9), -.5, .5)
                    velocities[eye, row, col] = offset*stride/(focal_px*dt)
                    confidence[eye, row, col] = min(1., margin/.3)*max(0., (peak-.65)/.35)
        self.result = dict(angular_rad_s=velocities.tolist(), confidence=confidence.tolist(),
                           valid_fraction=float(np.mean(confidence > .1)), units='rad/s (image axes)')
        return self.result
