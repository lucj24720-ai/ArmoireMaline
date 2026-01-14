# Architecture ArmoireMaline

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Web)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Upload    │  │   Options   │  │      Résultats          │  │
│  │  Référence  │  │  Seuil      │  │  - Image annotée        │  │
│  │  Actuelle   │  │  Taille min │  │  - Nb outils manquants  │  │
│  └──────┬──────┘  │  Alignement │  │  - Liste des zones      │  │
│         │         └─────────────┘  └─────────────────────────┘  │
└─────────┼───────────────────────────────────────────────────────┘
          │ HTTP POST (Base64 images)
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BACKEND (Flask)                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    API REST (app.py)                        ││
│  │  POST /api/analyze  - Analyse deux images                   ││
│  │  GET  /api/health   - État du service                       ││
│  │  PUT  /api/settings - Paramètres                            ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────────┐│
│  │                  PIPELINE DE DÉTECTION                      ││
│  │                                                             ││
│  │  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  ││
│  │  │  Alignement │ → │ Différence   │ → │ Détection       │  ││
│  │  │  (ORB/SIFT) │   │ (Seuillage)  │   │ (Contours)      │  ││
│  │  └─────────────┘   └──────────────┘   └─────────────────┘  ││
│  │                                                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐│
│  │  simple_detector.py  │  │  ai_detector.py (optionnel)      ││
│  │  - OpenCV classique  │  │  - YOLO (détection d'objets)     ││
│  │  - Rapide            │  │  - SAM (segmentation)            ││
│  │  - Pas de GPU requis │  │  - Nécessite GPU recommandé      ││
│  └──────────────────────┘  └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Algorithme de détection (Solution Simple)

### Étape 1: Alignement des images

```
Image Référence          Image Actuelle
      │                        │
      ▼                        ▼
┌─────────────────────────────────────────┐
│         Détection de points clés        │
│              (ORB ou SIFT)              │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│      Matching des descripteurs          │
│         (BFMatcher + Lowe ratio)        │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│       Calcul de l'homographie           │
│           (RANSAC, seuil 5px)           │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│    Application de la transformation     │
│          (warpPerspective)              │
└─────────────────────────────────────────┘
```

### Étape 2: Calcul des différences

```
Image Référence              Image Alignée
(prétraitée)                 (prétraitée)
      │                           │
      │    CLAHE + GaussianBlur   │
      ▼                           ▼
   ┌──────────────────────────────────┐
   │      Différence absolue          │
   │   diff = |ref - current|         │
   │   (BGR + Grayscale combinés)     │
   └───────────────┬──────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────┐
   │      Seuillage binaire           │
   │      threshold = 30              │
   └───────────────┬──────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────┐
   │   Opérations morphologiques      │
   │   - Ouverture (supprime bruit)   │
   │   - Fermeture (comble trous)     │
   │   - Dilatation (agrandi zones)   │
   └───────────────────────────────────┘
```

### Étape 3: Détection des zones manquantes

```
       Masque binaire
             │
             ▼
┌─────────────────────────────────┐
│    Détection de contours        │
│    (findContours, EXTERNAL)     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Filtrage par aire            │
│    500 < aire < 100000 px²      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Filtrage par ratio           │
│    aspect_ratio < 10            │
│    (évite lignes fines)         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Fusion des chevauchements    │
│    (IoU > 0.3 → merge)          │
└────────────────┬────────────────┘
                 │
                 ▼
         Liste des outils
           manquants
```

## Technologies utilisées

### Backend
| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Serveur | Flask 3.0 | API REST |
| Vision | OpenCV 4.9 | Traitement d'images |
| Arrays | NumPy | Manipulation de données |
| IA (opt) | PyTorch + YOLO | Détection avancée |

### Frontend
| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Structure | HTML5 | Interface |
| Style | CSS3 (Flexbox/Grid) | Mise en page |
| Logique | JavaScript ES6+ | Interactivité |
| Images | Canvas API / Base64 | Manipulation |

## Flux de données

```
┌─────────┐     Base64      ┌─────────┐     numpy.array    ┌──────────┐
│ Browser │ ───────────────▶│ Flask   │ ─────────────────▶ │ OpenCV   │
│ (JS)    │                 │ (JSON)  │                    │ Detector │
└─────────┘                 └─────────┘                    └──────────┘
     ▲                           │                              │
     │                           │                              │
     │      JSON Response        │      Result Dict             │
     └───────────────────────────┴──────────────────────────────┘

Format de la réponse:
{
  "success": true,
  "missing_count": 2,
  "missing_tools": [
    {"x": 165, "y": 85, "width": 70, "height": 200, "confidence": 0.87},
    {"x": 460, "y": 105, "width": 80, "height": 180, "confidence": 0.92}
  ],
  "result_image": "data:image/jpeg;base64,...",
  "alignment": {"success": true, "num_matches": 245}
}
```

## Optimisations possibles

### Performance
- [ ] Cache des images prétraitées
- [ ] Réduction de résolution pour l'alignement
- [ ] Traitement asynchrone (Celery)
- [ ] GPU acceleration (CUDA OpenCV)

### Précision
- [ ] Entraînement YOLO sur dataset d'outils
- [ ] Multi-scale detection
- [ ] Ensemble de méthodes (voting)
- [ ] Post-processing avec règles métier

### UX
- [ ] Mode caméra temps réel
- [ ] Application mobile (React Native)
- [ ] Notifications push
- [ ] Historique des comparaisons
