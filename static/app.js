document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('resultsContainer');
    const embeddingResults = document.getElementById('embeddingResults');
    const classifierResults = document.getElementById('classifierResults');
    const resetBtn = document.getElementById('resetBtn');
    
    // File size limit (5MB)
    const MAX_FILE_SIZE = 5 * 1024 * 1024;
    
    // Setup event listeners
    fileInput.addEventListener('change', handleFileInputChange);
    setupDragAndDrop();
    
    // Reset button
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            fileInput.value = '';
            preview.innerHTML = '';
            resultsContainer.classList.add('hidden');
            window.scrollTo(0, 0);
        });
    }
    
    // Handle file selection from input
    function handleFileInputChange() {
        if (this.files.length) {
            const file = this.files[0];
            if (validateFile(file)) {
                handleFile(file);
            }
        }
    }
    
    // Setup drag and drop functionality
    function setupDragAndDrop() {
        // Prevent default behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        // Add visual feedback
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle the drop
        dropArea.addEventListener('drop', handleDrop, false);
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        
        if (file && validateFile(file)) {
            handleFile(file);
        }
    }
    
    // Validate the file
    function validateFile(file) {
        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            alert(`File too large. Maximum size is ${formatSize(MAX_FILE_SIZE)}.`);
            return false;
        }
        
        // Check file type
        if (!file.type.match('image.*')) {
            alert('Please select an image file.');
            return false;
        }
        
        return true;
    }
    
    // Process the selected file
    function handleFile(file) {
        showPreview(file);
        analyzeImage(file);
    }
    
    // Show image preview
    function showPreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            preview.innerHTML = `
                <div class="preview-container">
                    <img src="${e.target.result}" class="preview-image" alt="Selected image">
                    <p class="file-info">${file.name} (${formatSize(file.size)})</p>
                </div>
            `;
        };
        
        reader.readAsDataURL(file);
    }
    
    // Analyze the image
    async function analyzeImage(file) {
        // Show loading, hide results
        loading.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('image', file);
            
            // Send the request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display results
            displayResults(data);
            
            // Show results container
            resultsContainer.classList.remove('hidden');
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing image: ' + error.message);
        } finally {
            // Hide loading indicator
            loading.classList.add('hidden');
        }
    }
    
    // Display the results
    function displayResults(data) {
        // Embedding results
        embeddingResults.innerHTML = data.embedding_similarity.map(result => 
            createResultItemHTML(result.font, result.similarity)
        ).join('');
        
        // Classifier results
        classifierResults.innerHTML = data.classifier_predictions.map(result => 
            createResultItemHTML(result.font, result.probability)
        ).join('');
    }
    
    function createResultItemHTML(fontName, score) {
        const percentage = (score * 100).toFixed(1);
        
        // Create a temporary element to safely escape the font name
        const tempDiv = document.createElement('div');
        tempDiv.textContent = fontName;
        const safeFontName = tempDiv.innerHTML;
        
        return `
            <div class="result-item">
                <span class="font-name">${safeFontName}</span>
                <span class="score">${percentage}%</span>
                <div class="font-sample">
                    The quick brown fox jumps over the lazy dog
                </div>
                <button class="copy-btn" onclick="navigator.clipboard.writeText('${safeFontName}').then(() => { this.textContent = 'Copied!'; setTimeout(() => { this.textContent = 'Copy font name'; }, 1500); })">Copy font name</button>
            </div>
        `;
    }
    
    // Format file size
    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
});