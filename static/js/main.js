// static/js/script.js

// Global variables
let stream = null;
let currentFacingMode = 'user'; // 'user' or 'environment'
let uploadedFilePath = null;

// DOM elements
const dropzone = document.getElementById('dropzone');
const fileUpload = document.getElementById('file-upload');
const cameraModal = document.getElementById('camera-modal');
const cameraPreview = document.getElementById('camera-preview');
const switchCameraBtn = document.getElementById('switch-camera');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');
const previewImage = document.getElementById('preview-image');
const diagnosisResults = document.getElementById('diagnosis-results');
const progressBar = document.getElementById('progress-bar');
const viewDetailsContainer = document.getElementById('view-details-container');
const viewDetailsBtn = document.getElementById('view-details-btn');
const model2Results = document.getElementById('model2-results');
const model2ResultsContent = document.getElementById('model2-results-content');

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processImageFile(file);
    }
}

// Handle drop event
function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
        processImageFile(file);
    }
}

// Process image file
function processImageFile(file) {
    // Check file type and size
    if (!file.type.match('image.*')) {
        alert('Please select an image file (JPG, PNG)');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB limit');
        return;
    }

    // Create FormData and upload the file
    const formData = new FormData();
    formData.append('file', file);

    // Show preview immediately
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        startAnalysis(formData);
    };
    reader.readAsDataURL(file);
}

// Start analysis process
function startAnalysis(formData) {
    // Show processing section
    processingSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    viewDetailsContainer.classList.add('hidden');
    model2Results.classList.add('hidden');

    // If formData is provided, upload the file first
    if (formData) {
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadedFilePath = data.filepath;
                simulateAnalysis();
            } else {
                alert('Error uploading image: ' + data.error);
                processingSection.classList.add('hidden');
            }
        })
        .catch(error => {
            console.error('Error uploading image:', error);
            alert('Error uploading image');
            processingSection.classList.add('hidden');
        });
    }
}

// Simulate analysis with progress bar
function simulateAnalysis() {
    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 100) progress = 100;
        progressBar.style.width = `${progress}%`;

        if (progress === 100) {
            clearInterval(interval);
            setTimeout(() => {
                // Call the analyze endpoint
                fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ filepath: uploadedFilePath })
                })
                .then(response => response.json())
                .then(data => {
                    processingSection.classList.add('hidden');
                    showResults(data);
                })
                .catch(error => {
                    console.error('Error analyzing image:', error);
                    alert('Error analyzing image');
                    processingSection.classList.add('hidden');
                });
            }, 500);
        }
    }, 200);
}

// Show results from initial analysis
function showResults(data) {
    if (data['hasDisease']['label'] === 1) {
        diagnosisResults.innerHTML = '';

        diagnosisResults.innerHTML = `
            <div class="p-4 bg-red-50 rounded-lg border border-red-100">
                <div class="flex items-center justify-between mb-2">
                    <h4 class="font-medium text-red-700">Phát Hiện Bất Thường</h4>
                    <span class="px-2 py-1 bg-red-100 text-red-800 text-xs font-semibold rounded">Warning</span>
                </div>
            </div>
        `;

        viewDetailsContainer.classList.remove('hidden');

        resultsSection.classList.remove('hidden');
    } else {
        diagnosisResults.innerHTML = '';

        diagnosisResults.innerHTML = `
            <div class="p-4 bg-green-50 rounded-lg border border-green-100">
                <div class="flex items-center justify-between mb-2">
                    <h4 class="font-medium text-green-700">Không Phát Hiện Bất Thường</h4>
                    <span class="px-2 py-1 bg-green-100 text-green-800 text-xs font-semibold rounded">Safe</span>
                </div>
            </div>
        `;

        resultsSection.classList.remove('hidden');
    }
}

// Run detailed analysis
function runDetailedAnalysis() {
    viewDetailsBtn.disabled = true;
    viewDetailsBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...';

    fetch('/detailed-analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filepath: uploadedFilePath })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showDetailedResults(data);
        } else {
            alert('Error in detailed analysis: ' + data.error);
        }

        // Reset button state
        viewDetailsBtn.disabled = false;
        viewDetailsBtn.innerHTML = '<i class="fas fa-search-plus mr-2"></i> Xem Phân Tích Chi Tiết';
    })
    .catch(error => {
        console.error('Error in detailed analysis:', error);
        alert('Error in detailed analysis');

        // Reset button state
        viewDetailsBtn.disabled = false;
        viewDetailsBtn.innerHTML = '<i class="fas fa-search-plus mr-2"></i> Xem Phân Tích Chi Tiết';
    });
}

// Show detailed analysis results
function showDetailedResults(data) {
    model2ResultsContent.innerHTML = '';

    let diseasesHTML = `
        <div class="p-4 bg-purple-50 rounded-lg border border-purple-100">
            <h4 class="font-medium text-purple-700 mb-2">Bệnh Đã Phát Hiện</h4>
            <div class="space-y-4">
    `;

    data.diseases[0].forEach(disease => {
        let severityColor;
        if (disease.probability > 70) {
            severityColor = 'red';
        } else if (disease.probability > 30) {
            severityColor = 'yellow';
        } else {
            severityColor = 'orange';
        }

        diseasesHTML += `
            <div class="p-3 bg-white rounded-lg border border-gray-200">
                <div class="flex justify-between items-center mb-1">
                    <h5 class="font-medium text-gray-700">${disease.name}</h5>
                    <span class="px-2 py-1 bg-${severityColor}-100 text-${severityColor}-800 text-xs font-semibold rounded">${disease.probability}% probability</span>
                </div>
            </div>
        `;
    });

    diseasesHTML += `
            </div>
        </div>
    `;

    // Combine and display
    model2ResultsContent.innerHTML = diseasesHTML;

    // Show model2 results
    model2Results.classList.remove('hidden');

    // Scroll to the detailed results
    model2Results.scrollIntoView({ behavior: 'smooth' });
}