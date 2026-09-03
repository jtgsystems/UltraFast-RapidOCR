import numpy as np
import mss
from typing import Tuple, Optional

class ScreenCapturer:
    """Ultra-fast zero-copy desktop screen capture via mss."""
    def __init__(self):
        self.sct = mss.mss()
        
    def capture_monitor(self, monitor_idx: int = 1) -> np.ndarray:
        """Capture entire monitor by index (1 = primary)."""
        monitors = self.sct.monitors
        if monitor_idx >= len(monitors):
            monitor_idx = 0
        sct_img = self.sct.grab(monitors[monitor_idx])
        # Convert to BGR numpy array
        return np.array(sct_img)[:, :, :3]

    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Capture specific bounding region on screen."""
        bbox = {"top": y, "left": x, "width": width, "height": height}
        sct_img = self.sct.grab(bbox)
        return np.array(sct_img)[:, :, :3]
