"""
Solution Simple - Détection d'outils manquants avec OpenCV classique.

Algorithme:
1. Alignement des images (homographie)
2. Prétraitement (réduction du bruit, normalisation)
3. Calcul de la différence absolue
4. Seuillage adaptatif
5. Opérations morphologiques
6. Détection et filtrage des contours
7. Génération du résultat visuel
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from image_alignment import ImageAligner, enhance_image_for_alignment


@dataclass
class MissingTool:
    """Représente un outil manquant détecté."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    area: int


class SimpleToolDetector:
    """
    Détecteur d'outils manquants basé sur la comparaison d'images.

    Cette approche compare pixel par pixel l'image de référence
    avec l'image actuelle pour identifier les zones de différence.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 100000,
        threshold: int = 30,
        alignment_method: str = "orb"
    ):
        """
        Args:
            min_area: Aire minimale (pixels²) pour considérer une zone
            max_area: Aire maximale pour éviter les faux positifs géants
            threshold: Seuil de différence (0-255)
            alignment_method: "orb" ou "sift" pour l'alignement
        """
        self.min_area = min_area
        self.max_area = max_area
        self.threshold = threshold
        self.aligner = ImageAligner(method=alignment_method)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Prétraitement de l'image pour réduire le bruit
        et normaliser les conditions d'éclairage.
        """
        # Réduction du bruit
        denoised = cv2.GaussianBlur(img, (5, 5), 0)

        # Conversion en LAB pour séparer luminosité et couleur
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Égalisation adaptative de l'histogramme (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_equalized = clahe.apply(l)

        # Reconstruction de l'image
        lab_equalized = cv2.merge([l_equalized, a, b])
        result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

        return result

    def compute_difference(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> np.ndarray:
        """
        Calcule la différence entre deux images.

        Utilise une combinaison de différence absolue sur les canaux BGR
        et une différence sur l'image en niveaux de gris pour robustesse.
        """
        # Différence en niveaux de gris
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(gray_ref, gray_cur)

        # Différence sur chaque canal couleur
        diff_bgr = cv2.absdiff(reference, current)
        diff_bgr_max = np.max(diff_bgr, axis=2)

        # Combinaison des deux différences (OR logique)
        combined = np.maximum(diff_gray, diff_bgr_max)

        return combined

    def create_mask(self, difference: np.ndarray) -> np.ndarray:
        """
        Crée un masque binaire à partir de l'image de différence.

        Applique un seuillage puis des opérations morphologiques
        pour nettoyer le masque.
        """
        # Seuillage
        _, binary = cv2.threshold(
            difference, self.threshold, 255, cv2.THRESH_BINARY
        )

        # Opérations morphologiques pour nettoyer
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Ouverture pour supprimer le bruit
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

        # Fermeture pour combler les trous
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)

        # Dilatation pour agrandir légèrement les zones
        cleaned = cv2.dilate(cleaned, kernel_small, iterations=2)

        return cleaned

    def find_missing_tools(self, mask: np.ndarray) -> List[MissingTool]:
        """
        Trouve les zones correspondant aux outils manquants.

        Détecte les contours et filtre par aire et forme.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        missing_tools = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filtrage par aire
            if area < self.min_area or area > self.max_area:
                continue

            # Calcul du rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)

            # Filtrage par ratio (évite les lignes trop fines)
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            if aspect_ratio > 10:  # Trop allongé = probablement du bruit
                continue

            # Calcul de la "solidité" (rapport aire/aire convexe)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Confiance basée sur la solidité et l'aire
            confidence = min(1.0, solidity * (area / self.max_area) * 2)

            missing_tools.append(MissingTool(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                area=area
            ))

        # Tri par aire décroissante
        missing_tools.sort(key=lambda t: t.area, reverse=True)

        return missing_tools

    def merge_overlapping(
        self,
        tools: List[MissingTool],
        overlap_threshold: float = 0.3
    ) -> List[MissingTool]:
        """
        Fusionne les détections qui se chevauchent.
        """
        if not tools:
            return []

        merged = []
        used = set()

        for i, tool1 in enumerate(tools):
            if i in used:
                continue

            current_box = [tool1.x, tool1.y, tool1.x + tool1.width, tool1.y + tool1.height]

            for j, tool2 in enumerate(tools[i+1:], start=i+1):
                if j in used:
                    continue

                box2 = [tool2.x, tool2.y, tool2.x + tool2.width, tool2.y + tool2.height]

                # Calcul de l'intersection
                inter_x1 = max(current_box[0], box2[0])
                inter_y1 = max(current_box[1], box2[1])
                inter_x2 = min(current_box[2], box2[2])
                inter_y2 = min(current_box[3], box2[3])

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    min_area = min(area1, area2)

                    if inter_area / min_area > overlap_threshold:
                        # Fusionner les boîtes
                        current_box[0] = min(current_box[0], box2[0])
                        current_box[1] = min(current_box[1], box2[1])
                        current_box[2] = max(current_box[2], box2[2])
                        current_box[3] = max(current_box[3], box2[3])
                        used.add(j)

            merged.append(MissingTool(
                x=current_box[0],
                y=current_box[1],
                width=current_box[2] - current_box[0],
                height=current_box[3] - current_box[1],
                confidence=tool1.confidence,
                area=(current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            ))
            used.add(i)

        return merged

    def detect(
        self,
        reference_img: np.ndarray,
        current_img: np.ndarray,
        align: bool = True
    ) -> Dict:
        """
        Détecte les outils manquants entre l'image de référence et l'actuelle.

        Args:
            reference_img: Image de l'armoire complète
            current_img: Image actuelle de l'armoire
            align: Si True, aligne automatiquement les images

        Returns:
            Dict contenant:
                - success: bool
                - missing_tools: List[MissingTool]
                - result_image: Image avec annotations
                - difference_image: Image de différence (debug)
                - mask_image: Masque binaire (debug)
                - alignment_info: Infos sur l'alignement
                - message: Message de statut
        """
        result = {
            "success": False,
            "missing_tools": [],
            "result_image": None,
            "difference_image": None,
            "mask_image": None,
            "alignment_info": None,
            "message": ""
        }

        # Redimensionner si les images ont des tailles différentes
        if reference_img.shape != current_img.shape:
            current_img = cv2.resize(
                current_img,
                (reference_img.shape[1], reference_img.shape[0])
            )

        # Alignement des images
        aligned_current = current_img
        if align:
            aligned_current, align_info = self.aligner.align(reference_img, current_img)
            result["alignment_info"] = align_info

            if aligned_current is None:
                # Utiliser l'image non alignée si l'alignement échoue
                aligned_current = current_img
                result["message"] = f"Alignement échoué: {align_info['message']}. Utilisation de l'image brute."

        # Prétraitement
        ref_processed = self.preprocess(reference_img)
        cur_processed = self.preprocess(aligned_current)

        # Calcul de la différence
        difference = self.compute_difference(ref_processed, cur_processed)
        result["difference_image"] = difference

        # Création du masque
        mask = self.create_mask(difference)
        result["mask_image"] = mask

        # Détection des zones manquantes
        missing_tools = self.find_missing_tools(mask)

        # Fusion des détections qui se chevauchent
        missing_tools = self.merge_overlapping(missing_tools)

        result["missing_tools"] = missing_tools

        # Création de l'image annotée
        result_img = reference_img.copy()
        for tool in missing_tools:
            # Rectangle rouge autour de la zone manquante
            cv2.rectangle(
                result_img,
                (tool.x, tool.y),
                (tool.x + tool.width, tool.y + tool.height),
                (0, 0, 255),  # Rouge en BGR
                3
            )
            # Fond semi-transparent
            overlay = result_img.copy()
            cv2.rectangle(
                overlay,
                (tool.x, tool.y),
                (tool.x + tool.width, tool.y + tool.height),
                (0, 0, 255),
                -1
            )
            cv2.addWeighted(overlay, 0.2, result_img, 0.8, 0, result_img)

        result["result_image"] = result_img
        result["success"] = True
        result["message"] = f"{len(missing_tools)} outil(s) manquant(s) détecté(s)"

        return result


def create_comparison_view(
    reference: np.ndarray,
    current: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    """
    Crée une vue comparative côte à côte des trois images.
    """
    h, w = reference.shape[:2]

    # Redimensionner pour l'affichage
    scale = min(1.0, 600 / w)
    new_w, new_h = int(w * scale), int(h * scale)

    ref_small = cv2.resize(reference, (new_w, new_h))
    cur_small = cv2.resize(current, (new_w, new_h))
    res_small = cv2.resize(result, (new_w, new_h))

    # Ajouter des labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(ref_small, "Reference", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(cur_small, "Actuelle", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(res_small, "Resultat", (10, 30), font, 1, (0, 255, 0), 2)

    # Assemblage horizontal
    comparison = np.hstack([ref_small, cur_small, res_small])

    return comparison


# Exemple d'utilisation
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python simple_detector.py <reference.jpg> <current.jpg>")
        sys.exit(1)

    ref_path, cur_path = sys.argv[1], sys.argv[2]

    # Chargement des images
    reference = cv2.imread(ref_path)
    current = cv2.imread(cur_path)

    if reference is None or current is None:
        print("Erreur: Impossible de charger les images")
        sys.exit(1)

    # Détection
    detector = SimpleToolDetector(min_area=500, threshold=30)
    result = detector.detect(reference, current)

    print(f"Résultat: {result['message']}")
    for i, tool in enumerate(result['missing_tools']):
        print(f"  Zone {i+1}: position=({tool.x}, {tool.y}), taille={tool.width}x{tool.height}")

    # Affichage
    cv2.imshow("Resultat", result['result_image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
