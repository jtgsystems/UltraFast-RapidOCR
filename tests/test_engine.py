import unittest
import numpy as np
import cv2
from hyper_ocr.engine import HyperOCREngine

class TestHyperOCREngine(unittest.TestCase):
    def setUp(self):
        self.engine = HyperOCREngine()

    def test_empty_image(self):
        empty = np.array([])
        res, lat = self.engine.recognize(empty)
        self.assertEqual(res, [])
        self.assertEqual(lat, 0.0)

    def test_synthetic_text_recognition(self):
        img = np.zeros((200, 600, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)
        cv2.putText(img, "JTG SYSTEMS", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        results, lat = self.engine.recognize(img)
        self.assertTrue(len(results) > 0)
        found_text = " ".join([r["text"] for r in results]).upper()
        self.assertIn("JTG", found_text)

    def test_temporal_cache(self):
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)
        cv2.putText(img, "CACHE TEST", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # First call (miss)
        res1, lat1 = self.engine.recognize(img, use_cache=True)
        # Second call (hit)
        res2, lat2 = self.engine.recognize(img, use_cache=True)
        
        self.assertEqual(res1, res2)
        self.assertLess(lat2, 10.0)  # Sub-10ms cache retrieval

if __name__ == "__main__":
    unittest.main()
