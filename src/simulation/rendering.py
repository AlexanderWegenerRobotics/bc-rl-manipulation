import cv2
import numpy as np
import mujoco
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.sim import Simulation


class Renderer:
    def __init__(self, sim: 'Simulation', config: dict):
        self.sim      = sim
        self.enabled  = config.get('enabled', True)

        cam_cfg           = config.get('camera', {})
        self.width        = cam_cfg.get('width',   640)
        self.height       = cam_cfg.get('height',  480)
        self.fps          = cam_cfg.get("fps", 30)
        self.lookat       = cam_cfg.get("lookat", [0.3, 0.0, 0.4])
        self.azimuth      = cam_cfg.get("azimuth", 135)
        self.elevation    = cam_cfg.get("elevation", -25)
        self.distance     = cam_cfg.get("distance", 1.8)
        self._window_name = config.get('window_name', 'Simulation')
        self.stop_request = False

        if self.enabled:
            self._renderer = mujoco.Renderer(sim.mj_model, height=self.height, width=self.width)
            self._cam      = self._build_camera()
        else:
            self._renderer = None
            self._cam      = None

    def _build_camera(self) -> mujoco.MjvCamera:
        """Construct a free MjvCamera from pos and look_at using spherical coordinates."""
        cam           = mujoco.MjvCamera()
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat    = self.lookat
        cam.distance  = self.distance
        cam.azimuth   = self.azimuth
        cam.elevation = self.elevation

        return cam

    def render(self) -> Optional[np.ndarray]:
        """Render current scene, show in OpenCV window, return BGR frame."""
        if not self.enabled or self._renderer is None:
            return None

        with self.sim._lock:
            self._renderer.update_scene(self.sim.mj_data, camera=self._cam)

        frame_rgb = self._renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow(self._window_name, frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_request = True

        return frame_bgr

    def close(self):
        """Close OpenCV window and release the MuJoCo renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        cv2.destroyAllWindows()


def make_renderer(sim: 'Simulation', config: dict) -> Renderer:
    """Construct a Renderer from config. Returns a disabled one if rendering.enabled is false."""
    return Renderer(sim, config)