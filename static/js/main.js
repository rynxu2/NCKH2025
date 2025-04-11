// // static/js/main.js
//
// // Camera variables
// let stream = null;
// let currentFacingMode = 'environment'; // Start with rear camera
// let mediaStreamConstraints = {
//     video: {
//         facingMode: currentFacingMode,
//         width: { ideal: 1280 },
//         height: { ideal: 720 }
//     }
// };
//
// // Check if model is loaded on page load
// window.addEventListener('DOMContentLoaded', async () => {
//     try {
//         const response = await fetch('/model-status');
//         const data = await response.json();
//
//         // Show model status
//         const headerEl = document.querySelector('header p');
//         if (headerEl) {
//             if (data.model_loaded) {
//                 headerEl.innerHTML += `<br><span class="text-green-600">Using ${data.model_type} for analysis</span>`;
//             } else {
//                 headerEl.innerHTML += `<br><span class="text-yellow-600">Using ${data.model_type} - CheXNet model not found</span>`;
//             }
//         }
//     } catch (err) {
//         console.error("Could not check model status:", err);
//     }
//
//     // Update switch button text initially
//     const switchBtn = document.getElementById('switch-camera');
//     if (switchBtn) {
//         switchBtn.innerHTML = `<i class="fas fa-sync-alt mr-2"></i>Switch to ${currentFacingMode === 'user' ? 'Rear' : 'Front'} Camera`;
//     }
// });
//
// // This function connects to the Flask backend for analysis
// async function analyzeXRay(imageElement) {
//     // Show processing section
//     document.getElementById('processing-section').classList.remove('hidden');
//
//     // Hide results section if it was previously shown
//     document.getElementById('results-section').classList.add('hidden');
//
//     // Start progress
//     let progress = 0;
//     const progressInterval = setInterval(() => {
//         progress += Math.random() * 5; // Slower progress
//         if (progress > 95) progress = 95; // Wait for actual completion
//         document.getElementById('progress-bar').style.width = `${progress}%`;
//     }, 200);
//
//     try {
//         // Prepare data to send to server
//         const canvas = document.createElement('canvas');
//         canvas.width = imageElement.width || imageElement.naturalWidth;
//         canvas.height = imageElement.height || imageElement.naturalHeight;
//         const ctx = canvas.getContext('2d');
//         ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
//
//         // Convert to blob
//         const blob = await new Promise(resolve =>
//             canvas.toBlob(resolve, 'image/jpeg', 0.95)
//         );
//
//         const formData = new FormData();
//         formData.append('xray_image', blob, 'xray.jpg');
//
//         // Send to server
//         const response = await fetch('/analyze', {
//             method: 'POST',
//             body: formData
//         });
//
//         if (!response.ok) {
//             throw new Error(`Server returned ${response.status}: ${response.statusText}`);
//         }
//
//         const data = await response.json();
//
//         // Complete progress bar
//         clearInterval(progressInterval);
//         document.getElementById('progress-bar').style.width = '100%';
//
//         // Small delay to show completed progress
//         setTimeout(() => {
//             document.getElementById('processing-section').classList.add('hidden');
//             showResultsFromBackend(imageElement.src, data.results, data.image_url);
//         }, 500);
//
//     } catch (error) {
//         console.error('Error analyzing X-ray:', error);
//
//         clearInterval(progressInterval);
//         document.getElementById('progress-bar').style.width = '100%';
//
//         setTimeout(() => {
//             document.getElementById('processing-section').classList.add('hidden');
//             showError(error.message);
//         }, 500);
//     }
// }
//
// function showError(errorMessage) {
//     // Show results section with error
//     document.getElementById('results-section').classList.remove('hidden');
//
//     // Set a placeholder image
//     document.getElementById('preview-image').src = '';
//
//     const resultsContainer = document.getElementById('diagnosis-results');
//     resultsContainer.innerHTML = `
//         <div class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
//             <h3 class="font-medium">Error Processing Image</h3>
//             <p>${errorMessage}</p>
//             <p class="mt-2">Please try again with a different image.</p>
//         </div>
//     `;
//
//     // Animate results in
//     setTimeout(() => {
//         document.querySelector('.result-card').classList.add('show');
//     }, 100);
// }
//
// function showResultsFromBackend(originalImageSrc, conditions, serverImageUrl) {
//     // Show results section
//     console.log(originalImageSrc, conditions, serverImageUrl)
//     document.getElementById('results-section').classList.remove('hidden');
//
//     // Set the preview image - use server URL if available, otherwise original source
//     document.getElementById('preview-image').src = serverImageUrl || originalImageSrc;
//
//     const resultsContainer = document.getElementById('diagnosis-results');
//     resultsContainer.innerHTML = '';
//
//     if (conditions && conditions.conditions.length > 0) {
//         conditions.conditions.forEach(condition => {
//             const resultItem = document.createElement('div');
//             resultItem.className = 'bg-gray-50 rounded-lg p-4';
//             resultItem.innerHTML = `
//                 <div class="flex justify-between items-center mb-2">
//                     <span class="font-medium text-gray-700">${condition.name}</span>
//                     <span class="text-sm font-semibold ${getProbabilityColor(condition.probability)}">${condition.probability}%</span>
//                 </div>
//                 <div class="w-full bg-gray-200 rounded-full h-2">
//                     <div class="h-2 rounded-full ${getProbabilityBarColor(condition.probability)}" style="width: ${condition.probability}%"></div>
//                 </div>
//             `;
//             resultsContainer.appendChild(resultItem);
//         });
//     } else {
//         resultsContainer.innerHTML = `
//             <div class="bg-gray-50 rounded-lg p-4">
//                 <p class="text-gray-700">No conditions detected or inconclusive results.</p>
//             </div>
//         `;
//     }
//
//     // Animate results in
//     setTimeout(() => {
//         document.querySelector('.result-card').classList.add('show');
//     }, 100);
// }
//
// function getProbabilityColor(probability) {
//     if (probability > 50) return 'text-red-600';
//     if (probability > 20) return 'text-yellow-600';
//     return 'text-green-600';
// }
//
// function getProbabilityBarColor(probability) {
//     if (probability > 50) return 'bg-red-500';
//     if (probability > 20) return 'bg-yellow-500';
//     return 'bg-green-500';
// }
//
// function handleFileSelect(event) {
//     const file = event.target.files[0];
//     console.log(file)
//     if (file) {
//         processImage(file);
//     }
// }
//
// function handleDrop(event) {
//     const file = event.dataTransfer.files[0];
//     if (file && file.type.match('image.*')) {
//         processImage(file);
//     }
// }
//
// function processImage(file) {
//     // Check file size (max 5MB)
//     if (file.size > 5 * 1024 * 1024) {
//         alert('File size exceeds 5MB limit. Please choose a smaller file.');
//         return;
//     }
//
//     const reader = new FileReader();
//     reader.onload = function(e) {
//         const image = new Image();
//         image.onload = function() {
//             analyzeXRay(image);
//         };
//         image.src = e.target.result;
//     };
//     reader.readAsDataURL(file);
// }
//
// // Camera functions
// function openCameraModal() {
//     const modal = document.getElementById('camera-modal');
//     modal.classList.add('open');
//     startCamera();
// }
//
// function closeCameraModal() {
//     const modal = document.getElementById('camera-modal');
//     modal.classList.remove('open');
//     stopCamera();
// }
//
// async function startCamera() {
//     try {
//         const video = document.getElementById('camera-preview');
//         stream = await navigator.mediaDevices.getUserMedia(mediaStreamConstraints);
//         video.srcObject = stream;
//     } catch (err) {
//         console.error("Error accessing camera:", err);
//         alert("Could not access the camera. Please make sure you have granted camera permissions.");
//         closeCameraModal();
//     }
// }
//
// function stopCamera() {
//     if (stream) {
//         stream.getTracks().forEach(track => track.stop());
//         const video = document.getElementById('camera-preview');
//         video.srcObject = null;
//         stream = null;
//     }
// }
//
// async function switchCamera() {
//     currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
//     mediaStreamConstraints.video.facingMode = currentFacingMode;
//
//     // Update switch button text
//     const switchBtn = document.getElementById('switch-camera');
//     switchBtn.innerHTML = `<i class="fas fa-sync-alt mr-2"></i>Switch to ${currentFacingMode === 'user' ? 'Rear' : 'Front'} Camera`;
//
//     // Restart camera with new constraints
//     stopCamera();
//     await startCamera();
// }
//
// function capturePhoto() {
//     const video = document.getElementById('camera-preview');
//     const canvas = document.createElement('canvas');
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     const ctx = canvas.getContext('2d');
//     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
//
//     // Create an image element from the canvas
//     const image = new Image();
//     image.src = canvas.toDataURL('image/jpeg');
//     image.onload = function() {
//         closeCameraModal();
//         analyzeXRay(image);
//     };
// }
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
    } else {
        // If no formData (for captured images), simulate analysis directly
        simulateAnalysis();
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
    }, 300);
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