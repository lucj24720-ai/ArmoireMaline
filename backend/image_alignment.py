"""
Module de recalage (alignement) d'images.

Utilise la détection de points clés et l'homographie pour aligner
deux images prises sous des angles légèrement différents.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageAligner:
    """
    Classe pour aligner deux images en utilisant la détection de features.

    Algorithme:
    1. Détection de points clés avec ORB (rapide) ou SIFT (précis)
    2. Calcul des descripteurs
    3. Matching des points entre les deux images
    4. Calcul de la matrice d'homographie (transformation perspective)
    5. Application de la transformation sur l'image à aligner
    """

    def __init__(self, method: str = "orb", max_features: int = 1000):
        """
        Args:
            method: "orb" (rapide) ou "sift" (plus précis mais plus lent)
            max_features: Nombre maximum de points clés à détecter
        """
        self.method = method
        self.max_features = max_features

        if method == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError(f"Méthode inconnue: {method}. Utilisez 'orb' ou 'sift'")

    def align(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray,
        min_matches: int = 10
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Aligne l'image cible sur l'image de référence.

        Args:
            reference_img: Image de référence (armoire complète)
            target_img: Image à aligner (armoire actuelle)
            min_matches: Nombre minimum de correspondances requises

        Returns:
            Tuple (image_alignée, infos) où infos contient:
                - success: bool
                - num_matches: nombre de correspondances trouvées
                - homography: matrice de transformation
                - message: message d'erreur si échec
        """
        info = {
            "success": False,
            "num_matches": 0,
            "homography": None,
            "message": ""
        }

        # Conversion en niveaux de gris
        gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # Détection des points clés et descripteurs
        kp_ref, desc_ref = self.detector.detectAndCompute(gray_ref, None)
        kp_target, desc_target = self.detector.detectAndCompute(gray_target, None)

        if desc_ref is None or desc_target is None:
            info["message"] = "Impossible de détecter des features dans une des images"
            return None, info

        if len(kp_ref) < min_matches or len(kp_target) < min_matches:
            info["message"] = f"Pas assez de points clés détectés (ref: {len(kp_ref)}, target: {len(kp_target)})"
            return None, info

        # Matching avec ratio test de Lowe
        matches = self.matcher.knnMatch(desc_target, desc_ref, k=2)

        # Filtrage des bons matchs (ratio test)
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        info["num_matches"] = len(good_matches)

        if len(good_matches) < min_matches:
            info["message"] = f"Pas assez de correspondances: {len(good_matches)} < {min_matches}"
            return None, info

        # Extraction des points correspondants
        src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calcul de l'homographie avec RANSAC pour robustesse
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            info["message"] = "Impossible de calculer l'homographie"
            return None, info

        info["homography"] = H

        # Application de la transformation
        h, w = reference_img.shape[:2]
        aligned_img = cv2.warpPerspective(target_img, H, (w, h))

        info["success"] = True
        info["message"] = f"Alignement réussi avec {len(good_matches)} correspondances"

        return aligned_img, info

    def visualize_matches(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray,
        max_display: int = 50
    ) -> np.ndarray:
        """
        Crée une visualisation des correspondances entre les deux images.
        Utile pour le debug et la validation.
        """
        gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        kp_ref, desc_ref = self.detector.detectAndCompute(gray_ref, None)
        kp_target, desc_target = self.detector.detectAndCompute(gray_target, None)

        if desc_ref is None or desc_target is None:
            return np.hstack([reference_img, target_img])

        matches = self.matcher.knnMatch(desc_target, desc_ref, k=2)

        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Limiter le nombre de matches affichés
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_display]

        result = cv2.drawMatches(
            target_img, kp_target,
            reference_img, kp_ref,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return result


def enhance_image_for_alignment(img: np.ndarray) -> np.ndarray:
    """
    Améliore une image pour faciliter l'alignement.
    Applique une égalisation d'histogramme adaptative (CLAHE).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced
