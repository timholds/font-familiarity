document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const preview = document.getElementById('preview');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const embeddingResults = document.getElementById('embeddingResults');
    const classifierResults = document.getElementById('classifierResults');
    const uploadArea = document.getElementById('uploadArea');
    
    // Hide results initially
    resultsContainer.classList.add('hidden');
    
    // Setup drag and drop
    setupDragAndDrop();
    
    // Handle file selection
    imageInput.addEventListener('change', handleFileSelection);
    
    // Handle form submission
    uploadForm.addEventListener('submit', handleFormSubmit);
    
    // Handle reset button
    resetBtn.addEventListener('click', resetUI);
    
    function setupDragAndDrop() {
        const dropArea = document.querySelector('.file-drop-area');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('highlight');
        }
        
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const file = dt.files[0];
            imageInput.files = dt.files;
            handleFileSelection();
        }
    }
    
    function handleFileSelection() {
        const file = imageInput.files[0];
        
        if (file) {
            // Enable analyze button
            analyzeBtn.disabled = false;
            
            // Clear previous preview
            preview.innerHTML = '';
            
            // Create object URL for preview
            const objectUrl = URL.createObjectURL(file);
            
            // Create preview image
            const img = new Image();
            img.onload = () => {
                console.log("Image loaded successfully:", img.width, "x", img.height);
            };
            
            img.onerror = (err) => {
                console.error("Error loading image:", err);
                preview.innerHTML = '<p class="error">Failed to load image preview</p>';
                analyzeBtn.disabled = true;
            };
            
            img.classList.add('preview-image');
            img.alt = "Preview of selected image";
            img.src = objectUrl;
            
            preview.appendChild(img);
            
            // Add file info
            const fileInfo = document.createElement('p');
            fileInfo.textContent = `${file.name} (${formatFileSize(file.size)})`;
            fileInfo.classList.add('file-info');
            preview.appendChild(fileInfo);
        } else {
            analyzeBtn.disabled = true;
            preview.innerHTML = '';
        }
    }
    
    async function handleFormSubmit(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            showError('Please select an image first');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        analyzeBtn.disabled = true;
        
        // Clear previous results
        embeddingResults.innerHTML = '';
        classifierResults.innerHTML = '';
        resultsContainer.classList.add('hidden');
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('image', file);
            
            // Send request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            // Handle non-200 responses
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Check for API error
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display results
            displayResults(data);
            
            // Show results container
            resultsContainer.classList.remove('hidden');
            
            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error:', error);
            showError(`Error: ${error.message || 'Failed to process image'}`);
        } finally {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    }
    
    function displayResults(data) {
        // Display embedding similarity results
        embeddingResults.innerHTML = data.embedding_similarity.map(result => 
            createResultItem(result.font, result.similarity)
        ).join('');
        
        // Display classifier predictions
        classifierResults.innerHTML = data.classifier_predictions.map(result => 
            createResultItem(result.font, result.probability)
        ).join('');
    }
    
    function createResultItem(fontName, score) {
        const percentage = (score * 100).toFixed(1);
        return `
            <div class="result-item">
                <div class="result-header">
                    <span class="font-name">${fontName}</span>
                    <span class="score">${percentage}%</span>
                </div>
                <div class="font-sample" style="font-family: '${fontName}', sans-serif">
                    The quick brown fox jumps over the lazy dog
                </div>
                <button class="copy-btn" data-font="${fontName}">Copy font name</button>
            </div>
        `;
    }
    
    function resetUI() {
        // Clear file input
        uploadForm.reset();
        
        // Clear preview
        preview.innerHTML = '';
        
        // Hide results
        resultsContainer.classList.add('hidden');
        
        // Disable analyze button
        analyzeBtn.disabled = true;
        
        // Focus on file input
        imageInput.focus();
    }
    
    function showError(message) {
        alert(message);
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }
    
    // Setup event delegation for copy buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('copy-btn')) {
            const fontName = e.target.getAttribute('data-font');
            navigator.clipboard.writeText(fontName)
                .then(() => {
                    // Show temporary success message
                    const originalText = e.target.textContent;
                    e.target.textContent = 'Copied!';
                    setTimeout(() => {
                        e.target.textContent = originalText;
                    }, 1500);
                })
                .catch(err => {
                    console.error('Failed to copy: ', err);
                });
        }
    });
});