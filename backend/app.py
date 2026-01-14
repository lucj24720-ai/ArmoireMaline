"""
API Flask pour ArmoireMaline.

Endpoints:
- POST /api/analyze : Compare deux images et retourne les outils manquants
- POST /api/upload/reference : Upload l'image de référence
- POST /api/upload/current : Upload l'image actuelle
- GET /api/health : Vérification de l'état du service
"""

import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

from simple_detector import SimpleToolDetector, create_comparison_view

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Instance du détecteur
detector = SimpleToolDetector(
    min_area=500,
    max_area=100000,
    threshold=30,
    alignment_method="orb"
)

# Stockage temporaire des sessions
sessions = {}


def allowed_file(filename: str) -> bool:
    """Vérifie si l'extension du fichier est autorisée."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_base64(img: np.ndarray) -> str:
    """Encode une image OpenCV en base64."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def decode_image_base64(data: str) -> np.ndarray:
    """Décode une image base64 en array numpy."""
    # Supprimer le préfixe data:image si présent
    if ',' in data:
        data = data.split(',')[1]

    img_bytes = base64.b64decode(data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@app.route('/')
def index():
    """Sert la page principale."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de vérification de santé."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@app.route('/api/session', methods=['POST'])
def create_session():
    """Crée une nouvelle session d'analyse."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "reference": None,
        "current": None
    }
    return jsonify({"session_id": session_id})


@app.route('/api/upload/reference', methods=['POST'])
def upload_reference():
    """Upload l'image de référence."""
    session_id = request.form.get('session_id')

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session invalide"}), 400

    if 'file' not in request.files:
        # Essayer avec base64
        data = request.json
        if data and 'image' in data:
            img = decode_image_base64(data['image'])
        else:
            return jsonify({"error": "Aucun fichier fourni"}), 400
    else:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Fichier vide"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Type de fichier non autorisé"}), 400

        # Lire l'image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Impossible de décoder l'image"}), 400

    # Sauvegarder dans la session
    sessions[session_id]["reference"] = img

    return jsonify({
        "success": True,
        "message": "Image de référence uploadée",
        "dimensions": {"width": img.shape[1], "height": img.shape[0]}
    })


@app.route('/api/upload/current', methods=['POST'])
def upload_current():
    """Upload l'image actuelle à comparer."""
    session_id = request.form.get('session_id')

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session invalide"}), 400

    if 'file' not in request.files:
        data = request.json
        if data and 'image' in data:
            img = decode_image_base64(data['image'])
        else:
            return jsonify({"error": "Aucun fichier fourni"}), 400
    else:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Fichier vide"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Type de fichier non autorisé"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Impossible de décoder l'image"}), 400

    sessions[session_id]["current"] = img

    return jsonify({
        "success": True,
        "message": "Image actuelle uploadée",
        "dimensions": {"width": img.shape[1], "height": img.shape[0]}
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyse les différences entre les deux images.

    Accepte soit un session_id, soit deux images en base64.
    """
    data = request.json or {}

    # Méthode 1: Via session
    session_id = data.get('session_id')
    if session_id and session_id in sessions:
        session = sessions[session_id]
        reference = session.get("reference")
        current = session.get("current")

        if reference is None:
            return jsonify({"error": "Image de référence manquante"}), 400
        if current is None:
            return jsonify({"error": "Image actuelle manquante"}), 400

    # Méthode 2: Images directement en base64
    elif 'reference' in data and 'current' in data:
        reference = decode_image_base64(data['reference'])
        current = decode_image_base64(data['current'])
    else:
        return jsonify({"error": "Fournissez session_id ou reference+current"}), 400

    # Paramètres optionnels
    options = data.get('options', {})
    min_area = options.get('min_area', 500)
    threshold = options.get('threshold', 30)
    align = options.get('align', True)

    # Configuration du détecteur
    detector.min_area = min_area
    detector.threshold = threshold

    # Analyse
    result = detector.detect(reference, current, align=align)

    # Préparer la réponse
    response = {
        "success": result["success"],
        "message": result["message"],
        "missing_count": len(result["missing_tools"]),
        "missing_tools": [
            {
                "x": t.x,
                "y": t.y,
                "width": t.width,
                "height": t.height,
                "confidence": t.confidence,
                "area": t.area
            }
            for t in result["missing_tools"]
        ]
    }

    # Ajouter les images si demandé
    if data.get('include_images', True):
        if result["result_image"] is not None:
            response["result_image"] = encode_image_base64(result["result_image"])
        if result["difference_image"] is not None:
            response["difference_image"] = encode_image_base64(result["difference_image"])
        if result["mask_image"] is not None:
            response["mask_image"] = encode_image_base64(result["mask_image"])

    # Infos d'alignement
    if result["alignment_info"]:
        response["alignment"] = {
            "success": result["alignment_info"]["success"],
            "num_matches": result["alignment_info"]["num_matches"],
            "message": result["alignment_info"]["message"]
        }

    return jsonify(response)


@app.route('/api/analyze/quick', methods=['POST'])
def analyze_quick():
    """
    Analyse rapide avec deux images envoyées directement.

    Attend un JSON avec:
    - reference: image base64
    - current: image base64
    """
    data = request.json

    if not data or 'reference' not in data or 'current' not in data:
        return jsonify({"error": "reference et current requis"}), 400

    try:
        reference = decode_image_base64(data['reference'])
        current = decode_image_base64(data['current'])
    except Exception as e:
        return jsonify({"error": f"Erreur de décodage: {str(e)}"}), 400

    result = detector.detect(reference, current)

    return jsonify({
        "success": result["success"],
        "message": result["message"],
        "missing_count": len(result["missing_tools"]),
        "result_image": encode_image_base64(result["result_image"]) if result["result_image"] is not None else None
    })


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Retourne les paramètres actuels du détecteur."""
    return jsonify({
        "min_area": detector.min_area,
        "max_area": detector.max_area,
        "threshold": detector.threshold,
        "alignment_method": detector.aligner.method
    })


@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Met à jour les paramètres du détecteur."""
    data = request.json

    if 'min_area' in data:
        detector.min_area = int(data['min_area'])
    if 'max_area' in data:
        detector.max_area = int(data['max_area'])
    if 'threshold' in data:
        detector.threshold = int(data['threshold'])

    return jsonify({"success": True, "message": "Paramètres mis à jour"})


# Nettoyage des sessions expirées
def cleanup_old_sessions(max_age_hours: int = 1):
    """Supprime les sessions plus vieilles que max_age_hours."""
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(hours=max_age_hours)

    to_delete = []
    for sid, session in sessions.items():
        created = datetime.fromisoformat(session["created_at"])
        if created < cutoff:
            to_delete.append(sid)

    for sid in to_delete:
        del sessions[sid]


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    ArmoireMaline API                      ║
    ║                                                           ║
    ║  Détection d'outils manquants dans une armoire            ║
    ║                                                           ║
    ║  Endpoints:                                               ║
    ║  - GET  /              → Interface web                    ║
    ║  - POST /api/analyze   → Analyser deux images             ║
    ║  - GET  /api/health    → État du service                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    app.run(host='0.0.0.0', port=5000, debug=True)
