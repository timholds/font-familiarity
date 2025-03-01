document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('dropArea');
    const imageInput = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const embeddingResults = document.getElementById('embeddingResults');
    const classifierResults = document.getElementById('classifierResults');
    const resetBtn = document.getElementById('resetBtn');
    
    // Constants for validation
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
    const VALID_FILE_TYPES = ['image/jpeg', 'image/png', 'image/gif'];
    
    // Setup drag and drop
    setupDragAndDrop();
    
    // Handle file selection from input
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file && validateFile(file)) {
            processSelectedFile(file);
        }
    });
    
    // Handle reset button
    resetBtn.addEventListener('click', function() {
        // Clear file input
        imageInput.value = '';
        
        // Clear preview
        preview.innerHTML = '';
        
        // Hide results
        resultsContainer.classList.add('hidden');
        
        // Scroll back to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    function setupDragAndDrop() {
        // Prevent default behavior for these events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop area when dragging over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        // Remove highlight when dragging stops
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('highlight');
        }
        
        // Handle dropped files
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const file = dt.files[0]; // Get the first file
            
            if (file && validateFile(file)) {
                processSelectedFile(file);
            }
        }
    }
    
    function processSelectedFile(file) {
        showPreview(file);
        analyzeImage(file);
    }
    
    function validateFile(file) {
        // Check file type
        if (!VALID_FILE_TYPES.includes(file.type)) {
            showError('Please select a valid image file (JPEG, PNG, or GIF)');
            return false;
        }
        
        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            showError(`File size exceeds limit (${formatFileSize(MAX_FILE_SIZE)})`);
            return false;
        }
        
        return true;
    }
    
    function showPreview(file) {
        // Clear previous preview
        preview.innerHTML = '';
        
        // Create object URL for preview
        const objectUrl = URL.createObjectURL(file);
        
        // Create preview container
        const previewContainer = document.createElement('div');
        previewContainer.className = 'preview-container';
        
        // Create preview image
        const img = new Image();
        img.onload = function() {
            // Image loaded successfully
            URL.revokeObjectURL(objectUrl); // Clean up the URL
        };
        
        img.onerror = function() {
            preview.innerHTML = '<p class="error">Failed to load image preview</p>';
            URL.revokeObjectURL(objectUrl);
        };
        
        img.classList.add('preview-image');
        img.alt = "Preview of selected image";
        img.src = objectUrl;
        
        // Create file info label
        const fileInfo = document.createElement('p');
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        fileInfo.classList.add('file-info');
        
        // Append elements
        previewContainer.appendChild(img);
        previewContainer.appendChild(fileInfo);
        preview.appendChild(previewContainer);
    }
    
    async function analyzeImage(file) {
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        
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
                throw new Error(`Server error: ${response.status}`);
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
            // Log detailed error to console, show generic message to user
            console.error('Error:', error);
            showError('An error occurred while processing the image');
        } finally {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
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
        
        // Add event listeners to copy buttons
        document.querySelectorAll('.copy-btn').forEach(btn => {
            btn.addEventListener('click', handleCopyClick);
        });
    }
    
    function createResultItem(fontName, score) {
        const percentage = (score * 100).toFixed(1);
        
        // Sanitize font name for safe DOM insertion
        const sanitizedFontName = document.createTextNode(fontName).textContent;
        
        return `
            <div class="result-item">
                <div class="result-header">
                    <span class="font-name">${sanitizedFontName}</span>
                    <span class="score">${percentage}%</span>
                </div>
                <div class="font-sample">
                    The quick brown fox jumps over the lazy dog
                </div>
                <button class="copy-btn" data-font="${sanitizedFontName}">Copy font name</button>
            </div>
        `;
    }
    
    function handleCopyClick(e) {
        const fontName = e.target.getAttribute('data-font');
        
        // Check for Clipboard API support
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(fontName)
                .then(() => {
                    // Show temporary success message
                    const originalText = e.target.textContent;
                    e.target.textContent = 'Copied!';
                    setTimeout(() => {
                        e.target.textContent = originalText;
                    }, 1500);
                })
                .catch(() => {
                    // Fallback for clipboard failures
                    promptToCopyManually(fontName);
                });
        } else {
            // Fallback for browsers without clipboard support
            promptToCopyManually(fontName);
        }
    }
    
    function promptToCopyManually(text) {
        // Create a temporary input element
        const tempInput = document.createElement('input');
        document.body.appendChild(tempInput);
        tempInput.value = text;
        tempInput.select();
        
        // Alert the user to use keyboard shortcut
        alert('Press Ctrl+C/Cmd+C to copy the font name');
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(tempInput);
        }, 100);
    }
    
    function showError(message) {
        alert(message);
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
});