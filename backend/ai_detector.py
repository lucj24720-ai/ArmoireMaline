"""
Solution Avancée - Détection d'outils manquants avec Intelligence Artificielle.

Cette solution utilise des modèles de deep learning pour:
1. Segmenter les objets individuels (SAM - Segment Anything Model)
2. Ou détecter les outils avec YOLO (You Only Look Once)

Avantages:
- Plus robuste aux variations de lumière
- Peut identifier le type d'outil
- Meilleure précision globale
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class DetectedTool:
    """Représente un outil détecté par l'IA."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_name: str
    mask: Optional[np.ndarray] = None


class AIToolDetector:
    """
    Détecteur d'outils basé sur l'IA.

    Supporte deux modes:
    - YOLO: Détection d'objets rapide
    - SAM: Segmentation précise

    Note: Nécessite l'installation des dépendances IA:
    pip install torch torchvision ultralytics segment-anything
    """

    def __init__(
        self,
        model_type: str = "yolo",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Args:
            model_type: "yolo" ou "sam"
            model_path: Chemin vers le modèle pré-entraîné
            confidence_threshold: Seuil de confiance (0-1)
            device: "cuda", "cpu" ou "auto"
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = device

        # Les modèles seront chargés à la demande
        self._model_loaded = False
        self._model_path = model_path

    def _load_model(self):
        """Charge le modèle IA (lazy loading)."""
        if self._model_loaded:
            return

        if self.model_type == "yolo":
            self._load_yolo()
        elif self.model_type == "sam":
            self._load_sam()
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")

        self._model_loaded = True

    def _load_yolo(self):
        """Charge le modèle YOLO."""
        try:
            from ultralytics import YOLO

            if self._model_path and os.path.exists(self._model_path):
                self.model = YOLO(self._model_path)
            else:
                # Utilise YOLOv8 pré-entraîné sur COCO
                # Pour les outils, un fine-tuning serait idéal
                self.model = YOLO("yolov8n.pt")
                print("Info: Utilisation du modèle YOLO générique. "
                      "Pour de meilleurs résultats, entraînez un modèle sur vos outils.")

        except ImportError:
            raise ImportError(
                "ultralytics non installé. Installez avec: pip install ultralytics"
            )

    def _load_sam(self):
        """Charge le modèle SAM (Segment Anything)."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model_type = "vit_h"  # Plus précis mais plus lent
            if self._model_path:
                checkpoint = self._model_path
            else:
                # Télécharger depuis: https://github.com/facebookresearch/segment-anything
                checkpoint = "models/sam_vit_h_4b8939.pth"

            if not os.path.exists(checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint SAM non trouvé: {checkpoint}\n"
                    "Téléchargez-le depuis: https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )

            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)

            self.model = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=100
            )

        except ImportError:
            raise ImportError(
                "segment_anything non installé. Installez avec: pip install segment-anything"
            )

    def detect_tools_yolo(self, img: np.ndarray) -> List[DetectedTool]:
        """
        Détecte les outils avec YOLO.

        Note: Le modèle YOLO standard détecte des objets génériques.
        Pour détecter spécifiquement des outils, il faut:
        1. Créer un dataset annoté de vos outils
        2. Fine-tuner YOLO sur ce dataset
        """
        self._load_model()

        # Classes d'outils pertinentes dans COCO
        tool_classes = {
            # Ces classes n'existent pas dans COCO standard
            # Mais pourraient être ajoutées avec un modèle personnalisé
            "hammer", "screwdriver", "wrench", "pliers", "drill",
            "saw", "scissors", "knife"
        }

        results = self.model(img, conf=self.confidence_threshold, verbose=False)
        detected = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                detected.append(DetectedTool(
                    x=x1, y=y1,
                    width=x2 - x1, height=y2 - y1,
                    confidence=conf,
                    class_name=class_name
                ))

        return detected

    def detect_tools_sam(self, img: np.ndarray) -> List[DetectedTool]:
        """
        Segmente tous les objets avec SAM.

        SAM est un modèle de segmentation "zero-shot" qui peut
        segmenter n'importe quel objet sans entraînement spécifique.
        """
        self._load_model()

        # Convertir BGR -> RGB pour SAM
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = self.model.generate(img_rgb)
        detected = []

        for mask_data in masks:
            mask = mask_data["segmentation"]
            bbox = mask_data["bbox"]  # x, y, w, h
            score = mask_data["predicted_iou"]

            if score < self.confidence_threshold:
                continue

            detected.append(DetectedTool(
                x=int(bbox[0]), y=int(bbox[1]),
                width=int(bbox[2]), height=int(bbox[3]),
                confidence=score,
                class_name="object",  # SAM ne classifie pas
                mask=mask.astype(np.uint8) * 255
            ))

        return detected

    def detect_tools(self, img: np.ndarray) -> List[DetectedTool]:
        """Détecte les outils selon le type de modèle configuré."""
        if self.model_type == "yolo":
            return self.detect_tools_yolo(img)
        elif self.model_type == "sam":
            return self.detect_tools_sam(img)
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")

    def compare_detections(
        self,
        ref_detections: List[DetectedTool],
        cur_detections: List[DetectedTool],
        iou_threshold: float = 0.5
    ) -> List[DetectedTool]:
        """
        Compare les détections pour trouver les outils manquants.

        Utilise l'IoU (Intersection over Union) pour matcher les objets.

        Args:
            ref_detections: Outils détectés dans l'image de référence
            cur_detections: Outils détectés dans l'image actuelle
            iou_threshold: Seuil IoU pour considérer un match

        Returns:
            Liste des outils présents dans ref mais pas dans cur
        """
        missing = []

        for ref_tool in ref_detections:
            found_match = False
            best_iou = 0

            ref_box = [ref_tool.x, ref_tool.y,
                       ref_tool.x + ref_tool.width,
                       ref_tool.y + ref_tool.height]

            for cur_tool in cur_detections:
                cur_box = [cur_tool.x, cur_tool.y,
                           cur_tool.x + cur_tool.width,
                           cur_tool.y + cur_tool.height]

                iou = self._calculate_iou(ref_box, cur_box)
                best_iou = max(best_iou, iou)

                if iou >= iou_threshold:
                    found_match = True
                    break

            if not found_match:
                missing.append(ref_tool)

        return missing

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calcule l'Intersection over Union entre deux boîtes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_missing(
        self,
        reference_img: np.ndarray,
        current_img: np.ndarray
    ) -> Dict:
        """
        Pipeline complet de détection des outils manquants.

        Args:
            reference_img: Image de l'armoire complète
            current_img: Image actuelle

        Returns:
            Dict avec les résultats de détection
        """
        result = {
            "success": False,
            "missing_tools": [],
            "ref_detections": [],
            "cur_detections": [],
            "result_image": None,
            "message": ""
        }

        try:
            # Détection dans l'image de référence
            ref_detections = self.detect_tools(reference_img)
            result["ref_detections"] = ref_detections

            # Détection dans l'image actuelle
            cur_detections = self.detect_tools(current_img)
            result["cur_detections"] = cur_detections

            # Comparaison
            missing = self.compare_detections(ref_detections, cur_detections)
            result["missing_tools"] = missing

            # Création de l'image annotée
            result_img = reference_img.copy()

            # Dessiner tous les outils de référence en vert
            for tool in ref_detections:
                cv2.rectangle(
                    result_img,
                    (tool.x, tool.y),
                    (tool.x + tool.width, tool.y + tool.height),
                    (0, 255, 0), 2
                )

            # Dessiner les outils manquants en rouge
            for tool in missing:
                cv2.rectangle(
                    result_img,
                    (tool.x, tool.y),
                    (tool.x + tool.width, tool.y + tool.height),
                    (0, 0, 255), 3
                )
                # Overlay semi-transparent
                overlay = result_img.copy()
                cv2.rectangle(
                    overlay,
                    (tool.x, tool.y),
                    (tool.x + tool.width, tool.y + tool.height),
                    (0, 0, 255), -1
                )
                cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0, result_img)

            result["result_image"] = result_img
            result["success"] = True
            result["message"] = (
                f"{len(missing)} outil(s) manquant(s) sur {len(ref_detections)} détecté(s)"
            )

        except Exception as e:
            result["message"] = f"Erreur: {str(e)}"

        return result


class HybridDetector:
    """
    Détecteur hybride combinant les approches classique et IA.

    Utilise la détection IA quand disponible, sinon fallback
    sur la méthode classique.
    """

    def __init__(self, prefer_ai: bool = True):
        self.prefer_ai = prefer_ai
        self._ai_available = self._check_ai_availability()

        # Import conditionnel
        from simple_detector import SimpleToolDetector
        self.simple_detector = SimpleToolDetector()

        if self._ai_available and prefer_ai:
            self.ai_detector = AIToolDetector(model_type="yolo")

    def _check_ai_availability(self) -> bool:
        """Vérifie si les dépendances IA sont installées."""
        try:
            import torch
            from ultralytics import YOLO
            return True
        except ImportError:
            return False

    def detect(
        self,
        reference_img: np.ndarray,
        current_img: np.ndarray
    ) -> Dict:
        """
        Détecte les outils manquants avec la meilleure méthode disponible.
        """
        if self._ai_available and self.prefer_ai:
            try:
                result = self.ai_detector.detect_missing(reference_img, current_img)
                result["method"] = "ai"
                return result
            except Exception as e:
                print(f"Fallback vers méthode simple: {e}")

        result = self.simple_detector.detect(reference_img, current_img)
        result["method"] = "simple"
        return result


# Guide pour créer un modèle YOLO personnalisé pour les outils
TRAINING_GUIDE = """
=== Guide d'entraînement d'un modèle YOLO pour vos outils ===

1. COLLECTE DE DONNÉES:
   - Photographiez chaque outil sous différents angles
   - Minimum 50-100 images par type d'outil
   - Variez l'éclairage et les arrière-plans

2. ANNOTATION:
   - Utilisez LabelImg ou Roboflow pour annoter
   - Créez une classe par type d'outil
   - Format YOLO: classe x_center y_center width height (normalisés)

3. STRUCTURE DU DATASET:
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml

4. FICHIER data.yaml:
   train: ./images/train
   val: ./images/val
   nc: 10  # nombre de classes
   names: ['hammer', 'screwdriver', 'wrench', ...]

5. ENTRAÎNEMENT:
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # modèle de base
   model.train(data='data.yaml', epochs=100, imgsz=640)

6. UTILISATION:
   ai_detector = AIToolDetector(
       model_type="yolo",
       model_path="runs/detect/train/weights/best.pt"
   )
"""


if __name__ == "__main__":
    print(TRAINING_GUIDE)
