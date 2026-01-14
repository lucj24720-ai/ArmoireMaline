# ArmoireMaline - Détection d'outils manquants

Application de vision par ordinateur pour détecter les outils manquants dans une armoire.

## 🎯 Fonctionnalités

- **Photo de référence** : Capture de l'état complet de l'armoire
- **Comparaison automatique** : Détection des différences entre deux états
- **Visualisation** : Entourage des zones où des outils sont manquants
- **Résumé** : Nombre d'outils manquants détectés

## 🏗️ Architecture

```
ArmoireMaline/
├── backend/
│   ├── app.py                 # API Flask principale
│   ├── simple_detector.py     # Solution OpenCV classique
│   ├── ai_detector.py         # Solution avec IA (YOLO/SAM)
│   ├── image_alignment.py     # Recalage d'images
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/                    # Modèles IA (à télécharger)
├── tests/
│   └── test_images/
└── README.md
```

## 🔧 Solutions proposées

### Solution 1 : OpenCV Classique (Simple)

**Algorithme :**
1. **Alignement** : Recalage des images avec détection de points clés (ORB/SIFT)
2. **Différence** : Calcul de la différence absolue entre les images
3. **Seuillage** : Binarisation pour isoler les changements significatifs
4. **Détection de contours** : Identification des zones manquantes
5. **Filtrage** : Élimination des faux positifs par taille/forme

**Avantages :**
- Rapide et léger
- Pas besoin de GPU
- Fonctionne hors-ligne

**Inconvénients :**
- Sensible aux variations de lumière
- Pas de reconnaissance d'objets

### Solution 2 : Intelligence Artificielle (Avancée)

**Algorithme :**
1. **Segmentation** : Utilisation de SAM (Segment Anything Model) ou YOLO
2. **Détection d'objets** : Identification de chaque outil individuellement
3. **Comparaison sémantique** : Matching des objets entre les deux images
4. **Rapport détaillé** : Liste des outils manquants avec leur type

**Avantages :**
- Robuste aux variations de lumière/angle
- Peut identifier le type d'outil manquant
- Meilleure précision

**Inconvénients :**
- Nécessite plus de ressources
- Temps de traitement plus long

## 🚀 Installation

```bash
# Cloner le projet
cd ArmoireMaline

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r backend/requirements.txt

# Lancer l'application
python backend/app.py
```

## 📱 Utilisation

1. Ouvrir http://localhost:5000 dans un navigateur
2. Uploader la photo de référence (armoire complète)
3. Uploader la photo actuelle
4. Cliquer sur "Analyser"
5. Visualiser les zones manquantes entourées en rouge

## 🔬 Algorithme détaillé

### Étape 1 : Alignement des images (Homographie)

```
Image Référence  →  Détection points clés (ORB)  →  Matching
       ↓                                              ↓
Image Actuelle   →  Détection points clés (ORB)  →  Calcul Homographie
                                                      ↓
                                              Image alignée
```

### Étape 2 : Détection des différences

```
Image Référence (alignée)
        ↓
   Différence absolue  →  Seuillage  →  Morphologie  →  Contours
        ↑
Image Actuelle
```

### Étape 3 : Filtrage et visualisation

```
Contours bruts  →  Filtrage par aire  →  Filtrage par ratio  →  Rectangles finaux
                   (min 500 pixels)      (évite les lignes)
```

## 📄 License

MIT License
