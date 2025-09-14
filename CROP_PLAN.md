# Simple Image Cropping - Minimal Implementation

## Goal
Let users click and drag to select the text area they want to analyze. Super simple, no fancy features.

## Total Implementation: ~50 lines of code

### Step 1: Add Cropper.js from CDN (2 lines)
Add to `templates/frontend.html` in the `<head>`:
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.1/cropper.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.1/cropper.min.js"></script>
```

### Step 2: Add a Crop Button (3 lines)
In `templates/frontend.html`, add after the preview div:
```html
<button id="cropBtn" class="hidden">Crop Selected Area</button>
```

### Step 3: Add the JavaScript (30 lines)
In `static/app.js`, add this:

```javascript
let cropper = null;

// Modify showPreview function to add crop option
function showPreview(file) {
    // ... existing preview code ...
    
    const img = previewContainer.querySelector('img');
    
    // Initialize cropper on the preview image
    img.onload = function() {
        // Show crop button
        document.getElementById('cropBtn').classList.remove('hidden');
        
        // Initialize cropper
        cropper = new Cropper(img, {
            aspectRatio: NaN, // Free selection
            viewMode: 1,
            guides: true,
            center: true,
            highlight: true,
            background: true,
            autoCrop: true,
            movable: false,
            rotatable: false,
            scalable: false,
            zoomable: false
        });
    };
}

// Handle crop button click
document.getElementById('cropBtn').addEventListener('click', function() {
    if (cropper) {
        cropper.getCroppedCanvas().toBlob(function(blob) {
            const croppedFile = new File([blob], 'cropped.jpg', {type: 'image/jpeg'});
            
            // Destroy cropper
            cropper.destroy();
            cropper = null;
            
            // Hide crop button
            document.getElementById('cropBtn').classList.add('hidden');
            
            // Analyze the cropped image
            analyzeImage(croppedFile);
        });
    }
});
```

### Step 4: Style the Button (5 lines)
Add to `static/style.css`:
```css
#cropBtn {
    background: #4a6baf;
    color: white;
    padding: 10px 20px;
    margin: 10px auto;
    display: block;
}
```

## That's it! ✅

### How it works:
1. User uploads image → sees preview with crop overlay
2. User drags to select text area
3. User clicks "Crop Selected Area" 
4. Cropped image gets analyzed

### What this gives you:
- Drag to select crop area
- Visual feedback with grid overlay
- Mobile touch support (built-in)
- Responsive to all image sizes
- Professional crop interface

### Total changes:
- 2 lines HTML for CDN
- 3 lines HTML for button  
- 30 lines JavaScript
- 5 lines CSS
- **Total: ~40 lines of code**

### Time to implement: 30 minutes

## Even Simpler Alternative: Skip Button Approach

Don't want the crop button? Make it automatic:

```javascript
// After 3 seconds, automatically use whatever they selected
img.onload = function() {
    cropper = new Cropper(img, {
        // ... same options ...
    });
    
    setTimeout(() => {
        alert("Adjust the selection box around your text, then click OK");
        // When they click OK, use the selection
        cropper.getCroppedCanvas().toBlob(function(blob) {
            analyzeImage(new File([blob], 'cropped.jpg', {type: 'image/jpeg'}));
        });
    }, 500);
};
```

## Why Cropper.js instead of custom canvas?
- **15 minutes vs 3 hours** implementation time
- **40 lines vs 200+ lines** of code
- Mobile gestures work perfectly out of the box
- Handles edge cases (image bounds, minimum size, etc.)
- Looks professional immediately
- Well-tested by millions of users

## Can always remove it
If you don't like it, just remove the CDN links and the 30 lines of JS. Your app works exactly as before.