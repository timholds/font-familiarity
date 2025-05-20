# Optimizing OCR Text Detection for Large-Scale Font Dataset Processing

## The Problem: Sequential Processing Bottleneck

When building our font recognition system, we faced a significant performance bottleneck during dataset preparation. Our approach used the CRAFT (Character-Region Awareness For Text detection) model to detect individual characters in font samples, which allowed our neural network to learn character-level features for better font recognition.

However, the default CRAFT implementation was designed to process images one at a time, creating a major bottleneck in our pipeline. With hundreds of thousands of font images to process, this sequential approach was prohibitively slow:

1. Each image went through multiple CPU↔GPU data transfers
2. Forward passes through the neural network were performed one image at a time
3. Post-processing steps like polygon extraction ran on CPU sequentially

For a dataset with 700,000+ images, this sequential approach would take nearly 12 hours to complete!

## The Solution: Monkeypatching CRAFT for Batch Processing

Instead of rewriting the CRAFT library, I implemented a monkeypatching approach that added batch processing capabilities while maintaining compatibility with the original codebase. The key insight was that most of the processing could be parallelized across a batch of images by careful refactoring.

### 1. Adding Batch Polygon Detection

The key optimization was implementing a `get_batch_polygons` method that processed multiple images in a single pass. In the original CRAFT codebase, polygon detection happened sequentially:

```python
# Original sequential approach (simplified)
def get_polygons(self, image: Image.Image) -> List[List[List[int]]]:
    # Preprocess single image
    x, ratio_w, ratio_h = preprocess_image(np.array(image), self.canvas_size, self.mag_ratio)
    
    # Forward pass for single image
    score_text, score_link = self.get_text_map(x, ratio_w, ratio_h)
    
    # Post-processing for single image
    boxes, polys = getDetBoxes(score_text, score_link, ...)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    # Convert to desired format
    res = []
    for poly in polys:
        res.append(poly.astype(np.int32).tolist())
    return res
```

I implemented a batch version that operates on tensors directly, keeping data on the GPU throughout the process:

```python
def get_batch_polygons(self, batch_images: torch.Tensor, ratios_w: torch.Tensor, ratios_h: torch.Tensor):
    """Batch process pre-normalized images on GPU"""
    # Forward pass for entire batch
    with torch.no_grad():
        y, _ = self.net(batch_images)
        if self.refiner:
            y, _ = self.refiner(y, None)

    # Batch post-processing on GPU
    text_scores = y[..., 0]  # [B, H, W]
    link_scores = y[..., 1] if not self.refiner else y[..., 0]
    
    # Threshold maps on GPU
    text_mask = (text_scores > self.text_threshold)
    link_mask = (link_scores > self.link_threshold)
    combined_mask = text_mask & link_mask

    # Process each image in batch (still much faster than full sequential)
    batch_polys = []
    for b_idx in range(batch_images.size(0)):
        # Extract polygons with GPU-accelerated connected components
        # ... processing code ...
        batch_polys.append(polys)

    return batch_polys
```

### 2. Optimizing Preprocessing and Post-processing

I also refactored the preprocessing and post-processing steps to handle batches efficiently:

```python
def batch_preprocess_image_np(batch_images, canvas_size, mag_ratio):
    """Process a batch of images with vectorized operations where possible"""
    batch_size = len(batch_images)
    resized_images = []
    ratios_w = []
    ratios_h = []
    
    # Resize each image (could be parallelized further with multiprocessing)
    for i in range(batch_size):
        img_resized, target_ratio, _ = resize_aspect_ratio(
            batch_images[i], canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio
        
        resized_images.append(img_resized)
        ratios_w.append(ratio_w)
        ratios_h.append(ratio_h)
    
    # Stack images into a single batch tensor
    batch_resized = np.stack(resized_images, axis=0)
    
    # Vectorized normalization (much faster than processing one by one)
    batch_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    batch_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    
    batch_normalized = (batch_resized / 255.0 - batch_mean) / batch_std
    
    # Transpose from [B, H, W, C] to [B, C, H, W] for PyTorch
    batch_transposed = np.transpose(batch_normalized, (0, 3, 1, 2))
    
    return batch_transposed, ratios_w, ratios_h
```

### 3. DataLoader Integration for Prefetching

To further optimize the pipeline, I integrated PyTorch's DataLoader to prefetch and prepare batches in parallel:

```python
def preprocess_craft_optimized(data_dir, device="cuda", batch_size=32, resume=True, num_workers=None):
    """Optimized version using DataLoader prefetching"""
    # ...setup code...
    
    # Create dataset for leveraging DataLoader's prefetching
    temp_dataset = CharacterFontDataset(
        data_dir, 
        train=(mode == 'train'),
        use_precomputed_craft=False
    )
    
    # Create DataLoader with prefetching
    temp_loader = DataLoader(
        temp_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2  # Prefetch 2 batches
    )
    
    # Process batches efficiently
    for batch_idx, batch in enumerate(tqdm(temp_loader)):
        # ...processing code using batch operations...
```

## Results: 10x Performance Improvement

The optimizations resulted in dramatic speed improvements:

- **Original sequential approach**: ~12 hours for 700K images
- **Optimized batch approach**: ~1.2 hours for 700K images

This 10x speedup significantly improved our development workflow, allowing us to iterate faster on the character-level font recognition approach.

## Key Takeaways

1. **Keep data on the GPU**: Minimize CPU↔GPU transfers by operating on batches
2. **Vectorize operations**: Use numpy/PyTorch's vectorized operations instead of loops
3. **Use prefetching**: DataLoader's prefetching capabilities minimize idle GPU time
4. **Consider robustness**: For long-running processes, implement checkpointing and recovery
5. **Monkeypatching vs. rewriting**: Extending existing libraries through monkeypatching can be an efficient approach when full rewrites aren't practical

This optimization was crucial to be able to train anything in a reasonable time on my good ol A4500. 