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
    
    // Constants for validation
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
    const VALID_FILE_TYPES = ['image/jpeg', 'image/png', 'image/gif'];
    
    // Setup drag and drop
    setupDragAndDrop();
    
    // Handle file selection
    imageInput.addEventListener('change', handleFileSelection);
    
    // Handle form submission
    uploadForm.addEventListener('submit', handleFormSubmit);
    
    // Handle form reset
    uploadForm.addEventListener('reset', function() {
        // Additional cleanup beyond the form reset
        preview.innerHTML = '';
        resultsContainer.classList.add('hidden');
        analyzeBtn.disabled = true;
        
        // Clear any object URLs
        if (window.objectUrlToRevoke) {
            URL.revokeObjectURL(window.objectUrlToRevoke);
            window.objectUrlToRevoke = null;
        }
    });
    
    function setupDragAndDrop() {
        const dropArea = document.querySelector('.file-drop-area');
        
        // Handle drag events
        dropArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('highlight');
        });
        
        dropArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('highlight');
        });
        
        dropArea.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('highlight');
            
            // Handle the dropped files
            if (e.dataTransfer.files && e.dataTransfer.files.length) {
                // Setting files property directly works in modern browsers
                imageInput.files = e.dataTransfer.files;
                
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                imageInput.dispatchEvent(event);
            }
        });
    }
    
    function handleFileSelection() {
        const file = imageInput.files[0];
        
        // Clear any existing object URL
        if (window.objectUrlToRevoke) {
            URL.revokeObjectURL(window.objectUrlToRevoke);
            window.objectUrlToRevoke = null;
        }
        
        if (file) {
            // Validate file
            if (!validateFile(file)) {
                return;
            }
            
            // Enable analyze button
            analyzeBtn.disabled = false;
            
            // Clear previous preview
            preview.innerHTML = '';
            
            // Create object URL for preview
            const objectUrl = URL.createObjectURL(file);
            window.objectUrlToRevoke = objectUrl;
            
            // Create preview image
            const img = new Image();
            img.onload = function() {
                // Image loaded successfully
            };
            
            img.onerror = function() {
                preview.innerHTML = '<p class="error">Failed to load image preview</p>';
                analyzeBtn.disabled = true;
                URL.revokeObjectURL(objectUrl);
                window.objectUrlToRevoke = null;
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
    
    function validateFile(file) {
        // Check file type
        if (!VALID_FILE_TYPES.includes(file.type)) {
            showError('Please select a valid image file (JPEG, PNG, or GIF)');
            imageInput.value = ''; // Clear the file input
            return false;
        }
        
        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            showError(`File size exceeds limit (${formatFileSize(MAX_FILE_SIZE)})`);
            imageInput.value = ''; // Clear the file input
            return false;
        }
        
        return true;
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
            
            // Add CSRF token if available (from data attribute)
            const csrfToken = uploadForm.dataset.csrfToken;
            if (csrfToken) {
                formData.append('csrf_token', csrfToken);
                // Also set it as a header for APIs that expect it there
                const headers = {
                    'X-CSRFToken': csrfToken
                };
            }
            
            // Send request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
                headers: csrfToken ? { 'X-CSRFToken': csrfToken } : {}
            });
            
            // Handle non-200 responses
            if (!response.ok) {
                throw new Error('Server error occurred');
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
            // Only log detailed error to console, show generic message to user
            console.error('Error:', error);
            showError('An error occurred while processing the image');
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
        
        // Add event listeners to copy buttons
        document.querySelectorAll('.copy-btn').forEach(btn => {
            btn.addEventListener('click', handleCopyClick);
        });
    }
    
    function createResultItem(fontName, score) {
        const percentage = (score * 100).toFixed(1);
        
        // Sanitize font name for safe DOM insertion by creating a text node
        // and then getting its content (browser handles the sanitization)
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