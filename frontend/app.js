/**
 * ArmoireMaline - Frontend Application
 * Gestion de l'interface utilisateur et communication avec l'API
 */

// Configuration
const API_BASE = window.location.origin;

// État de l'application
const state = {
    referenceImage: null,
    currentImage: null,
    sessionId: null,
    isAnalyzing: false
};

// Éléments DOM
const elements = {
    referenceInput: document.getElementById('reference-input'),
    currentInput: document.getElementById('current-input'),
    referencePreview: document.getElementById('reference-preview'),
    currentPreview: document.getElementById('current-preview'),
    referenceDrop: document.getElementById('reference-drop'),
    currentDrop: document.getElementById('current-drop'),
    clearReference: document.getElementById('clear-reference'),
    clearCurrent: document.getElementById('clear-current'),
    analyzeBtn: document.getElementById('analyze-btn'),
    resultsSection: document.getElementById('results-section'),
    loading: document.getElementById('loading'),
    missingCount: document.getElementById('missing-count'),
    summary: document.getElementById('summary'),
    resultImage: document.getElementById('result-image'),
    differenceImage: document.getElementById('difference-image'),
    comparisonRef: document.getElementById('comparison-ref'),
    comparisonCur: document.getElementById('comparison-cur'),
    zonesList: document.getElementById('zones-list'),
    alignmentInfo: document.getElementById('alignment-info'),
    threshold: document.getElementById('threshold'),
    thresholdValue: document.getElementById('threshold-value'),
    minArea: document.getElementById('min-area'),
    minAreaValue: document.getElementById('min-area-value'),
    alignImages: document.getElementById('align-images'),
    tabs: document.querySelectorAll('.tab'),
    tabPanes: document.querySelectorAll('.tab-pane')
};

// Initialisation
function init() {
    setupDropZones();
    setupFileInputs();
    setupClearButtons();
    setupAnalyzeButton();
    setupOptions();
    setupTabs();
}

// Configuration des zones de drop
function setupDropZones() {
    [elements.referenceDrop, elements.currentDrop].forEach((dropZone, index) => {
        const isReference = index === 0;

        // Événements de drag
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');

            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleImageUpload(file, isReference);
            }
        });

        // Clic pour sélectionner
        dropZone.addEventListener('click', () => {
            const input = isReference ? elements.referenceInput : elements.currentInput;
            input.click();
        });
    });
}

// Configuration des inputs fichier
function setupFileInputs() {
    elements.referenceInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleImageUpload(file, true);
    });

    elements.currentInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleImageUpload(file, false);
    });
}

// Gestion de l'upload d'image
function handleImageUpload(file, isReference) {
    const reader = new FileReader();

    reader.onload = (e) => {
        const dataUrl = e.target.result;

        if (isReference) {
            state.referenceImage = dataUrl;
            elements.referencePreview.src = dataUrl;
            elements.referencePreview.classList.remove('hidden');
            elements.referenceDrop.querySelector('.drop-content').classList.add('hidden');
            elements.clearReference.classList.remove('hidden');
        } else {
            state.currentImage = dataUrl;
            elements.currentPreview.src = dataUrl;
            elements.currentPreview.classList.remove('hidden');
            elements.currentDrop.querySelector('.drop-content').classList.add('hidden');
            elements.clearCurrent.classList.remove('hidden');
        }

        updateAnalyzeButton();
    };

    reader.readAsDataURL(file);
}

// Configuration des boutons de suppression
function setupClearButtons() {
    elements.clearReference.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage(true);
    });

    elements.clearCurrent.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage(false);
    });
}

// Supprimer une image
function clearImage(isReference) {
    if (isReference) {
        state.referenceImage = null;
        elements.referencePreview.classList.add('hidden');
        elements.referenceDrop.querySelector('.drop-content').classList.remove('hidden');
        elements.clearReference.classList.add('hidden');
        elements.referenceInput.value = '';
    } else {
        state.currentImage = null;
        elements.currentPreview.classList.add('hidden');
        elements.currentDrop.querySelector('.drop-content').classList.remove('hidden');
        elements.clearCurrent.classList.add('hidden');
        elements.currentInput.value = '';
    }

    updateAnalyzeButton();
    elements.resultsSection.classList.add('hidden');
}

// Mise à jour du bouton d'analyse
function updateAnalyzeButton() {
    const canAnalyze = state.referenceImage && state.currentImage && !state.isAnalyzing;
    elements.analyzeBtn.disabled = !canAnalyze;
}

// Configuration du bouton d'analyse
function setupAnalyzeButton() {
    elements.analyzeBtn.addEventListener('click', analyze);
}

// Analyse des images
async function analyze() {
    if (!state.referenceImage || !state.currentImage) return;

    state.isAnalyzing = true;
    updateAnalyzeButton();
    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                reference: state.referenceImage,
                current: state.currentImage,
                include_images: true,
                options: {
                    threshold: parseInt(elements.threshold.value),
                    min_area: parseInt(elements.minArea.value),
                    align: elements.alignImages.checked
                }
            })
        });

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        displayResults(result);

    } catch (error) {
        console.error('Erreur d\'analyse:', error);
        alert(`Erreur lors de l'analyse: ${error.message}`);
    } finally {
        state.isAnalyzing = false;
        updateAnalyzeButton();
        showLoading(false);
    }
}

// Affichage des résultats
function displayResults(result) {
    elements.resultsSection.classList.remove('hidden');

    // Mise à jour du résumé
    const count = result.missing_count;
    elements.missingCount.textContent = count;

    // Style du résumé selon le nombre
    elements.summary.className = 'summary-card';
    if (count === 0) {
        elements.summary.classList.add('success');
        elements.summary.querySelector('.summary-icon').textContent = '✅';
    } else if (count <= 2) {
        elements.summary.classList.add('warning');
        elements.summary.querySelector('.summary-icon').textContent = '⚠️';
    } else {
        elements.summary.classList.add('danger');
        elements.summary.querySelector('.summary-icon').textContent = '🚨';
    }

    // Images résultats
    if (result.result_image) {
        elements.resultImage.src = `data:image/jpeg;base64,${result.result_image}`;
    }
    if (result.difference_image) {
        elements.differenceImage.src = `data:image/jpeg;base64,${result.difference_image}`;
    }

    // Images de comparaison
    elements.comparisonRef.src = state.referenceImage;
    elements.comparisonCur.src = state.currentImage;

    // Liste des zones
    elements.zonesList.innerHTML = '';
    if (result.missing_tools && result.missing_tools.length > 0) {
        result.missing_tools.forEach((tool, index) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>Zone ${index + 1}</span>
                <span class="zone-info">
                    Position: (${tool.x}, ${tool.y}) |
                    Taille: ${tool.width}x${tool.height}px
                </span>
            `;
            elements.zonesList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'Aucun outil manquant détecté';
        li.style.textAlign = 'center';
        li.style.color = 'var(--success-color)';
        elements.zonesList.appendChild(li);
    }

    // Info d'alignement
    if (result.alignment) {
        elements.alignmentInfo.classList.remove('hidden');
        elements.alignmentInfo.className = 'alignment-info ' +
            (result.alignment.success ? 'success' : 'failed');
        elements.alignmentInfo.querySelector('.alignment-status').textContent =
            result.alignment.success ? 'Alignement réussi' : 'Alignement échoué';
        elements.alignmentInfo.querySelector('.alignment-details').textContent =
            `(${result.alignment.num_matches} points de correspondance)`;
    } else {
        elements.alignmentInfo.classList.add('hidden');
    }

    // Scroll vers les résultats
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Configuration des options
function setupOptions() {
    elements.threshold.addEventListener('input', (e) => {
        elements.thresholdValue.textContent = e.target.value;
    });

    elements.minArea.addEventListener('input', (e) => {
        elements.minAreaValue.textContent = e.target.value;
    });
}

// Configuration des onglets
function setupTabs() {
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            // Mise à jour des onglets
            elements.tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Mise à jour des panneaux
            elements.tabPanes.forEach(pane => {
                pane.classList.remove('active');
                if (pane.id === `tab-${targetTab}`) {
                    pane.classList.add('active');
                }
            });
        });
    });
}

// Afficher/masquer le loading
function showLoading(show) {
    if (show) {
        elements.loading.classList.remove('hidden');
    } else {
        elements.loading.classList.add('hidden');
    }
}

// Démarrer l'application
document.addEventListener('DOMContentLoaded', init);
