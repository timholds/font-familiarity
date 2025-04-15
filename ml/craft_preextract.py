import os
import numpy as np
from tqdm import tqdm
from CRAFT import CRAFTModel
from dataset import load_npz_mmap, load_h5_dataset
import argparse
from PIL import Image

def preprocess_craft(data_dir, device="cuda", batch_size=32):
    """Preprocess CRAFT results for entire dataset"""
    for mode in ['train', 'test']:
        print(f"Processing {mode} set...")
        
        # Initialize CRAFT model - only once, outside the loop
        craft_model = CRAFTModel(
                cache_dir='weights/',
                device=device,
                use_refiner=True,
                fp16=(device == "cuda"),  # Use fp16 only on CUDA
                link_threshold=1.9,
                text_threshold=.5,
                low_text=.5,
            )
        
        # Load dataset
        h5_file = os.path.join(data_dir, f'{mode}.h5')
        npz_file = os.path.join(data_dir, f'{mode}.npz')
        
        if os.path.exists(h5_file):
            print(f"Loading H5 dataset from {h5_file}")
            images, labels, h5_file_handle = load_h5_dataset(h5_file)
            using_h5 = True
        elif os.path.exists(npz_file):
            print(f"Loading NPZ dataset from {npz_file}")
            images, labels = load_npz_mmap(npz_file)
            using_h5 = False
            h5_file_handle = None
        else:
            raise FileNotFoundError(f"No dataset file found at {h5_file} or {npz_file}")
        
        # Create output file
        output_file = os.path.join(data_dir, f'{mode}_craft_boxes.npz')
        
        # Process images in batches - sequential approach
        num_images = len(images)
        all_boxes = []
        
        for i in tqdm(range(0, num_images, batch_size)):
            batch_indices = range(i, min(i + batch_size, num_images))
            batch_images = [Image.fromarray(images[j].astype(np.uint8)) for j in batch_indices]
            
            # Process batch with CRAFT - sequential approach
            batch_boxes = []
            for img in batch_images:
                try:
                    # Use a try/except that could mimic the "CUDA error" scenario
                    # by falling back to CPU if there's a CUDA error
                    try:
                        polygons = craft_model.get_polygons(img)
                    except RuntimeError as e:
                        if "CUDA" in str(e) and device == "cuda":
                            print("CUDA error detected, falling back to CPU for this image")
                            # Create a temporary CPU model for fallback
                            cpu_model = CRAFTModel(
                                cache_dir='weights/',
                                device="cpu",
                                use_refiner=True,
                                fp16=False,  # Must be False for CPU
                                link_threshold=1.9,
                                text_threshold=.5,
                                low_text=.5,
                            )
                            polygons = cpu_model.get_polygons(img)
                        else:
                            # Re-raise if it's not a CUDA error
                            raise
                    
                    # Convert polygons to bounding boxes
                    boxes = []
                    for poly in polygons:
                        x_coords = [p[0] for p in poly]
                        y_coords = [p[1] for p in poly]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    batch_boxes.append(boxes)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    batch_boxes.append([])
            
            all_boxes.extend(batch_boxes)
        
        # Save incrementally to avoid losing progress
        if i % (batch_size * 10) == 0:
            np.savez_compressed(
                output_file + ".partial",
                boxes=np.array(all_boxes, dtype=object)
            )
            print(f"Saved partial progress ({len(all_boxes)} boxes) to {output_file}.partial")
        
        # Save final boxes to file
        np.savez_compressed(
            output_file,
            boxes=np.array(all_boxes, dtype=object)
        )
        
        print(f"Saved {len(all_boxes)} box sets to {output_file}")
        
        # Close H5 file if opened
        if using_h5 and h5_file_handle is not None:
            h5_file_handle.close()

if __name__ == "__main__":
    import argparse
    import multiprocessing
    parser = argparse.ArgumentParser(description="Preprocess CRAFT results for font dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    preprocess_craft(args.data_dir, device="cuda", batch_size=args.batch_size)