"""
Tests unitaires pour le détecteur d'outils manquants.
"""

import sys
import os
import unittest
import numpy as np
import cv2

# Ajouter le dossier backend au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from simple_detector import SimpleToolDetector, MissingTool
from image_alignment import ImageAligner, enhance_image_for_alignment


class TestImageAlignment(unittest.TestCase):
    """Tests pour le module d'alignement d'images."""

    def setUp(self):
        """Crée des images de test."""
        # Image de référence avec un pattern distinctif
        self.reference = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(self.reference, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(self.reference, (200, 100), (350, 200), (128, 128, 128), -1)
        cv2.circle(self.reference, (500, 300), 50, (200, 200, 200), -1)

        # Ajout de texture pour les features
        for i in range(0, 600, 20):
            for j in range(0, 400, 20):
                cv2.circle(self.reference, (i, j), 2, (100, 100, 100), -1)

        # Image légèrement décalée
        M = np.float32([[1, 0, 10], [0, 1, 5]])  # Translation
        self.shifted = cv2.warpAffine(self.reference, M, (600, 400))

    def test_aligner_creation(self):
        """Test la création de l'aligneur."""
        aligner_orb = ImageAligner(method="orb")
        self.assertIsNotNone(aligner_orb.detector)

        aligner_sift = ImageAligner(method="sift")
        self.assertIsNotNone(aligner_sift.detector)

    def test_align_identical_images(self):
        """Test l'alignement de deux images identiques."""
        aligner = ImageAligner(method="orb")
        aligned, info = aligner.align(self.reference, self.reference.copy())

        self.assertTrue(info["success"])
        self.assertIsNotNone(aligned)

    def test_align_shifted_image(self):
        """Test l'alignement d'une image décalée."""
        aligner = ImageAligner(method="orb")
        aligned, info = aligner.align(self.reference, self.shifted)

        # L'alignement devrait réussir
        self.assertTrue(info["success"])
        self.assertGreater(info["num_matches"], 10)

    def test_enhance_image(self):
        """Test l'amélioration d'image."""
        enhanced = enhance_image_for_alignment(self.reference)

        self.assertEqual(enhanced.shape, self.reference.shape)
        self.assertEqual(enhanced.dtype, self.reference.dtype)


class TestSimpleDetector(unittest.TestCase):
    """Tests pour le détecteur simple."""

    def setUp(self):
        """Crée des images de test."""
        # Image de référence : fond gris avec outils (rectangles blancs)
        self.reference = np.full((400, 600, 3), 100, dtype=np.uint8)

        # Simuler 3 outils
        cv2.rectangle(self.reference, (50, 50), (120, 180), (200, 200, 200), -1)
        cv2.rectangle(self.reference, (200, 80), (350, 160), (180, 180, 180), -1)
        cv2.rectangle(self.reference, (450, 200), (550, 350), (220, 220, 220), -1)

        # Image actuelle : un outil manquant (le deuxième)
        self.current = self.reference.copy()
        cv2.rectangle(self.current, (200, 80), (350, 160), (100, 100, 100), -1)

        self.detector = SimpleToolDetector(min_area=500, threshold=30)

    def test_detector_creation(self):
        """Test la création du détecteur."""
        detector = SimpleToolDetector()
        self.assertIsNotNone(detector.aligner)
        self.assertEqual(detector.min_area, 500)

    def test_preprocess(self):
        """Test le prétraitement."""
        processed = self.detector.preprocess(self.reference)

        self.assertEqual(processed.shape, self.reference.shape)
        self.assertEqual(processed.dtype, np.uint8)

    def test_compute_difference(self):
        """Test le calcul de différence."""
        diff = self.detector.compute_difference(self.reference, self.current)

        self.assertEqual(len(diff.shape), 2)  # Image en niveaux de gris
        self.assertTrue(np.max(diff) > 0)  # Il y a des différences

    def test_create_mask(self):
        """Test la création du masque."""
        diff = self.detector.compute_difference(self.reference, self.current)
        mask = self.detector.create_mask(diff)

        self.assertEqual(len(mask.shape), 2)
        self.assertTrue(np.max(mask) == 255)  # Masque binaire

    def test_find_missing_tools(self):
        """Test la détection des zones manquantes."""
        diff = self.detector.compute_difference(self.reference, self.current)
        mask = self.detector.create_mask(diff)
        missing = self.detector.find_missing_tools(mask)

        # Devrait détecter au moins une zone
        self.assertGreater(len(missing), 0)

    def test_detect_full_pipeline(self):
        """Test le pipeline complet de détection."""
        result = self.detector.detect(self.reference, self.current, align=False)

        self.assertTrue(result["success"])
        self.assertGreater(len(result["missing_tools"]), 0)
        self.assertIsNotNone(result["result_image"])
        self.assertIsNotNone(result["difference_image"])
        self.assertIsNotNone(result["mask_image"])

    def test_no_differences(self):
        """Test quand il n'y a pas de différences."""
        result = self.detector.detect(self.reference, self.reference.copy(), align=False)

        self.assertTrue(result["success"])
        self.assertEqual(len(result["missing_tools"]), 0)

    def test_merge_overlapping(self):
        """Test la fusion des détections qui se chevauchent."""
        tools = [
            MissingTool(x=10, y=10, width=50, height=50, confidence=0.9, area=2500),
            MissingTool(x=30, y=30, width=50, height=50, confidence=0.8, area=2500),  # Chevauchement
            MissingTool(x=200, y=200, width=50, height=50, confidence=0.7, area=2500)  # Séparé
        ]

        merged = self.detector.merge_overlapping(tools)

        # Les deux premiers devraient être fusionnés
        self.assertEqual(len(merged), 2)


class TestMissingToolDataclass(unittest.TestCase):
    """Tests pour la dataclass MissingTool."""

    def test_creation(self):
        """Test la création d'un MissingTool."""
        tool = MissingTool(x=10, y=20, width=100, height=50, confidence=0.95, area=5000)

        self.assertEqual(tool.x, 10)
        self.assertEqual(tool.y, 20)
        self.assertEqual(tool.width, 100)
        self.assertEqual(tool.height, 50)
        self.assertAlmostEqual(tool.confidence, 0.95)
        self.assertEqual(tool.area, 5000)


class TestEdgeCases(unittest.TestCase):
    """Tests des cas limites."""

    def setUp(self):
        self.detector = SimpleToolDetector()

    def test_different_image_sizes(self):
        """Test avec des images de tailles différentes."""
        ref = np.zeros((400, 600, 3), dtype=np.uint8)
        cur = np.zeros((300, 500, 3), dtype=np.uint8)

        # Devrait redimensionner automatiquement
        result = self.detector.detect(ref, cur, align=False)
        self.assertTrue(result["success"])

    def test_very_small_differences(self):
        """Test avec des différences très petites (sous le seuil)."""
        ref = np.full((400, 600, 3), 100, dtype=np.uint8)
        cur = np.full((400, 600, 3), 105, dtype=np.uint8)  # Petite différence

        result = self.detector.detect(ref, cur, align=False)
        # Ne devrait pas détecter de différences significatives
        self.assertEqual(len(result["missing_tools"]), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
