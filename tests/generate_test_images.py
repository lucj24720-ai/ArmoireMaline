"""
Génère des images de test pour ArmoireMaline.

Crée une armoire simulée avec des outils pour tester l'application.
"""

import cv2
import numpy as np
import os


def create_toolbox_reference():
    """
    Crée une image de référence d'une armoire à outils complète.

    Retourne une image 800x600 avec plusieurs outils simulés.
    """
    # Fond de l'armoire (gris foncé)
    img = np.full((600, 800, 3), 60, dtype=np.uint8)

    # Panneau perforé (motif de trous)
    for y in range(20, 580, 30):
        for x in range(20, 780, 30):
            cv2.circle(img, (x, y), 5, (40, 40, 40), -1)

    # Outil 1: Marteau (silhouette)
    # Tête
    cv2.rectangle(img, (50, 80), (110, 130), (180, 180, 180), -1)
    # Manche
    cv2.rectangle(img, (70, 130), (90, 250), (139, 90, 43), -1)

    # Outil 2: Tournevis
    # Manche
    cv2.ellipse(img, (200, 100), (30, 15), 0, 0, 360, (200, 50, 50), -1)
    # Tige
    cv2.rectangle(img, (195, 115), (205, 280), (150, 150, 150), -1)

    # Outil 3: Clé à molette
    cv2.rectangle(img, (300, 70), (400, 100), (170, 170, 170), -1)
    cv2.rectangle(img, (340, 100), (360, 260), (170, 170, 170), -1)

    # Outil 4: Pince
    # Corps
    cv2.ellipse(img, (500, 120), (40, 25), 0, 0, 360, (160, 160, 160), -1)
    # Manches
    cv2.rectangle(img, (470, 145), (490, 280), (100, 100, 100), -1)
    cv2.rectangle(img, (510, 145), (530, 280), (100, 100, 100), -1)

    # Outil 5: Clé plate
    cv2.ellipse(img, (650, 90), (25, 15), 0, 0, 360, (175, 175, 175), -1)
    cv2.rectangle(img, (640, 105), (660, 250), (175, 175, 175), -1)
    cv2.ellipse(img, (650, 265), (20, 12), 0, 0, 360, (175, 175, 175), -1)

    # Outil 6: Scie (bas de l'armoire)
    cv2.rectangle(img, (50, 350), (300, 370), (180, 180, 180), -1)
    # Dents de scie
    for x in range(55, 300, 10):
        cv2.line(img, (x, 370), (x + 5, 380), (180, 180, 180), 2)
    # Manche
    cv2.rectangle(img, (50, 320), (100, 380), (139, 90, 43), -1)

    # Outil 7: Mètre ruban
    cv2.rectangle(img, (400, 350), (500, 450), (255, 200, 0), -1)
    cv2.rectangle(img, (420, 370), (480, 430), (50, 50, 50), -1)

    # Outil 8: Niveau à bulle
    cv2.rectangle(img, (550, 400), (750, 430), (0, 180, 0), -1)
    cv2.ellipse(img, (650, 415), (30, 10), 0, 0, 360, (200, 255, 200), -1)

    # Outil 9: Petite clé (rangée du haut)
    cv2.ellipse(img, (720, 90), (15, 10), 0, 0, 360, (165, 165, 165), -1)
    cv2.rectangle(img, (712, 100), (728, 180), (165, 165, 165), -1)

    # Ajout d'une légère texture
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def create_toolbox_missing(reference, missing_tools):
    """
    Crée une image avec des outils manquants.

    Args:
        reference: Image de référence
        missing_tools: Liste de tuples (x1, y1, x2, y2) définissant les zones à "vider"

    Returns:
        Image avec les outils manquants (zones remplacées par le fond)
    """
    img = reference.copy()

    # Couleur du fond (panneau perforé)
    background_color = (60, 60, 60)

    for (x1, y1, x2, y2) in missing_tools:
        # Remplir la zone avec le fond
        cv2.rectangle(img, (x1, y1), (x2, y2), background_color, -1)

        # Redessiner les trous du panneau perforé
        for y in range(y1 + 15, y2, 30):
            for x in range(x1 + 15, x2, 30):
                if 20 <= x < 780 and 20 <= y < 580:
                    cv2.circle(img, (x, y), 5, (40, 40, 40), -1)

    return img


def add_lighting_variation(img, factor=1.1):
    """Simule une variation d'éclairage."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_slight_rotation(img, angle=1.5):
    """Ajoute une légère rotation."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def main():
    """Génère les images de test."""
    output_dir = os.path.dirname(__file__)
    images_dir = os.path.join(output_dir, 'test_images')
    os.makedirs(images_dir, exist_ok=True)

    print("Génération des images de test...")

    # 1. Image de référence (armoire complète)
    reference = create_toolbox_reference()
    cv2.imwrite(os.path.join(images_dir, 'reference.jpg'), reference)
    print("  - reference.jpg créée")

    # 2. Un outil manquant (le tournevis)
    missing_1 = create_toolbox_missing(reference, [
        (165, 85, 235, 285)  # Tournevis
    ])
    cv2.imwrite(os.path.join(images_dir, 'missing_1.jpg'), missing_1)
    print("  - missing_1.jpg créée (1 outil manquant)")

    # 3. Trois outils manquants
    missing_3 = create_toolbox_missing(reference, [
        (165, 85, 235, 285),    # Tournevis
        (460, 105, 540, 285),   # Pince
        (390, 340, 510, 460)    # Mètre ruban
    ])
    cv2.imwrite(os.path.join(images_dir, 'missing_3.jpg'), missing_3)
    print("  - missing_3.jpg créée (3 outils manquants)")

    # 4. Image avec variation de lumière
    missing_light = add_lighting_variation(missing_1, 1.15)
    cv2.imwrite(os.path.join(images_dir, 'missing_1_bright.jpg'), missing_light)
    print("  - missing_1_bright.jpg créée (avec variation de lumière)")

    # 5. Image légèrement tournée
    missing_rotated = add_slight_rotation(missing_1, 2)
    cv2.imwrite(os.path.join(images_dir, 'missing_1_rotated.jpg'), missing_rotated)
    print("  - missing_1_rotated.jpg créée (avec légère rotation)")

    # 6. Aucun outil manquant (juste variation de lumière)
    no_missing = add_lighting_variation(reference, 0.95)
    cv2.imwrite(os.path.join(images_dir, 'no_missing.jpg'), no_missing)
    print("  - no_missing.jpg créée (aucun outil manquant)")

    print(f"\nImages générées dans: {images_dir}")
    print("\nUtilisation recommandée:")
    print("  1. Comparer reference.jpg avec missing_1.jpg")
    print("  2. Comparer reference.jpg avec missing_3.jpg")
    print("  3. Tester la robustesse avec missing_1_bright.jpg et missing_1_rotated.jpg")


if __name__ == '__main__':
    main()
