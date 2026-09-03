import time
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional

def get_free_vram_mb() -> float:
    """Check free GPU VRAM in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return free_bytes / (1024 * 1024)
    except Exception:
        pass
    return 0.0

class HyperOCREngine:
    """
    SOTA 2026 Multi-Engine Accelerated OCR System.
    Features Adaptive VRAM Management, CUDA GPU -> ONNX CPU fail-safe fallback,
    and SIMD Temporal Frame Caching.
    """
    def __init__(self, prefer_gpu: bool = True, confidence_threshold: float = 0.4):
        self.prefer_gpu = prefer_gpu
        self.conf_thresh = confidence_threshold
        self.gpu_available = False
        self.reader = None
        self.rapid_engine = None
        
        # 1. Try initializing PyTorch CUDA if sufficient VRAM (>800MB)
        if prefer_gpu and get_free_vram_mb() > 800:
            try:
                import torch
                if torch.cuda.is_available():
                    import easyocr
                    self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                    self.gpu_available = True
            except Exception:
                self.gpu_available = False
                
        # 2. Always prepare RapidOCR ONNX fallback
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.rapid_engine = RapidOCR()
        except Exception:
            pass
            
        # Differential Frame Cache
        self._last_hash = None
        self._last_results = []

    def recognize(self, image: np.ndarray, use_cache: bool = True) -> Tuple[List[Dict[str, Any]], float]:
        """
        Recognize text in an image / screen buffer.
        Returns: (results_list, latency_ms)
        """
        if image is None or image.size == 0:
            return [], 0.0
            
        t0 = time.perf_counter()
        
        # Temporal frame cache check for video/screen streams
        if use_cache and image.ndim >= 2:
            h = hash(image[::16, ::16].tobytes())
            if h == self._last_hash and self._last_results:
                latency = (time.perf_counter() - t0) * 1000
                return self._last_results, latency
            self._last_hash = h
            
        results = []
        gpu_succeeded = False
        
        # 1. Try GPU Pipeline if free VRAM is healthy (>500MB)
        if self.gpu_available and self.reader and get_free_vram_mb() > 500:
            try:
                h, w = image.shape[:2]
                scale = 1.0
                max_dim = max(h, w)
                if max_dim > 1600:
                    scale = 1600.0 / max_dim
                    proc_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                else:
                    proc_img = image
                    
                raw_res = self.reader.readtext(proc_img)
                for bbox, text, conf in raw_res:
                    if conf >= self.conf_thresh:
                        box_list = [[int(pt[0] / scale), int(pt[1] / scale)] for pt in bbox]
                        results.append({
                            "text": text.strip(),
                            "confidence": float(conf),
                            "box": box_list
                        })
                gpu_succeeded = True
            except Exception:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gpu_succeeded = False
                
        # 2. Seamless Fast Fallback to RapidOCR ONNX
        if not gpu_succeeded and self.rapid_engine:
            raw_res, _ = self.rapid_engine(image, use_det=True, use_cls=False, use_rec=True)
            if raw_res:
                for box, text, score in raw_res:
                    conf = float(score) if isinstance(score, (int, float, str)) else 0.8
                    if conf >= self.conf_thresh:
                        results.append({
                            "text": text.strip(),
                            "confidence": conf,
                            "box": box
                        })
                        
        self._last_results = results
        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms
