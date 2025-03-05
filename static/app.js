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
    
    function showPreview(file) {
        // Clear previous preview
        preview.innerHTML = '';
        
        // Show the reset button
        document.getElementById('resetButtonContainer').classList.remove('hidden');
        
        // Create object URL for preview
        const objectUrl = URL.createObjectURL(file);
        
        // Create preview container
        const previewContainer = document.createElement('div');
        previewContainer.className = 'preview-container';
        
        // Create preview image
        const img = new Image();
        img.onload = function() {
            URL.revokeObjectURL(objectUrl);
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
        fileInfo.textContent = `${file.name} (${formatSize(file.size)})`;
        fileInfo.classList.add('file-info');
        
        // Append elements
        previewContainer.appendChild(img);
        previewContainer.appendChild(fileInfo);
        preview.appendChild(previewContainer);
    }
    
    // Update reset button handler
    resetBtn.addEventListener('click', function() {
        // Clear file input
        fileInput.value = '';
        
        // Clear preview
        preview.innerHTML = '';
        
        // Hide results
        resultsContainer.classList.add('hidden');
        
        // Hide reset button
        document.getElementById('resetButtonContainer').classList.add('hidden');
    });
    
    // Function to format font name from model format (lowercase_with_underscores) to display format (Title Case)
    function formatFontName(modelFontName) {
        // Convert underscores to spaces
        let formattedName = modelFontName.replace(/_/g, ' ');
        
        // Capitalize each word
        formattedName = formattedName.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
            
        // Handle special cases
        const specialCases = {
            'Pt Sans': 'PT Sans',
            'Pt Serif': 'PT Serif',
            'Ibm Plex': 'IBM Plex',
            'Dm Sans': 'DM Sans',
            'Dm Serif': 'DM Serif',
            'Eb Garamond': 'EB Garamond'
        };
        
        // Check if the formatted name starts with any special case key
        for (const [special, correct] of Object.entries(specialCases)) {
            if (formattedName.startsWith(special)) {
                formattedName = formattedName.replace(special, correct);
                break;
            }
        }
        
        return formattedName;
    }
    
    // Preload Google Fonts
    async function preloadGoogleFont(modelFontName) {
        // Get properly formatted name 
        const displayName = formatFontName(modelFontName);
        
        // Create Google Fonts URL
        const googleFontParam = displayName.replace(/\s+/g, '+');
        const fontUrl = `https://fonts.googleapis.com/css2?family=${googleFontParam}&display=swap`;
        
        // Check if we already loaded this font
        const existingLink = document.getElementById(`font-${googleFontParam}`);
        if (existingLink) {
            return displayName; // Font already requested
        }
        
        // Add link element
        const link = document.createElement('link');
        link.id = `font-${googleFontParam}`;
        link.rel = 'stylesheet';
        link.href = fontUrl;
        document.head.appendChild(link);
        
        console.log(`Loading font: ${displayName} (URL: ${fontUrl})`);
        
        // Return the display name
        return displayName;
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
            
            // Preload all fonts before displaying results
            const fontPromises = [];
            
            // Collect all unique fonts from both result sets
            const allFonts = new Set();
            
            data.embedding_similarity.forEach(result => allFonts.add(result.font));
            data.classifier_predictions.forEach(result => allFonts.add(result.font));
            
            // Preload all fonts in parallel
            for (const fontName of allFonts) {
                fontPromises.push(preloadGoogleFont(fontName));
            }
            
            // Wait for all fonts to load
            await Promise.all(fontPromises);
            
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
    
    function createResultItemHTML(modelFontName, score) {
        const percentage = (score * 100).toFixed(1);
        
        // Format the font name for display
        const displayFontName = formatFontName(modelFontName);
        
        // Create the font-family CSS value
        const fontFamily = `'${displayFontName}', sans-serif`;
        
        return `
            <div class="result-item">
                <div class="result-header">
                    <span class="font-name" style="font-family: ${fontFamily};">${displayFontName}</span>
                    <span class="score">${percentage}%</span>
                </div>
                <div class="font-sample" style="font-family: ${fontFamily};">
                    The quick brown fox jumps over the lazy dog 1234567890
                </div>
                <div class="font-actions">
                    <button class="copy-btn" onclick="navigator.clipboard.writeText('${displayFontName}').then(() => { this.textContent = 'Copied!'; setTimeout(() => { this.textContent = 'Copy font name'; }, 1500); })">Copy font name</button>
                    <a href="https://fonts.google.com/specimen/${displayFontName.replace(/\s+/g, '+')}" target="_blank" class="font-link">View on Google Fonts</a>
                </div>
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