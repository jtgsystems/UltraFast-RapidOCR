import unittest
import os
import cv2
import numpy as np
from hyper_ocr.video import VideoTextExtractor

class TestVideoExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_video = "/tmp/test_hyper_ocr_unit.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.test_video, fourcc, 10.0, (640, 360))
        
        # Write 20 frames with clean text
        for _ in range(20):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:] = (255, 255, 255)
            cv2.putText(frame, "JTG SYSTEMS 2026", (50, 180), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_video):
            try:
                os.remove(cls.test_video)
            except Exception:
                pass

    def test_video_text_extraction(self):
        extractor = VideoTextExtractor(sample_fps=2.0)
        timeline = extractor.extract_from_video(self.test_video)
        self.assertTrue(len(timeline) > 0)
        recognized = timeline[0]["text"].upper()
        self.assertTrue("JTG" in recognized or "2026" in recognized)

if __name__ == "__main__":
    unittest.main()
