import os
import numpy as np
from tqdm import tqdm
from CRAFT import CRAFTModel
from CRAFT.craft import init_CRAFT_model
from CRAFT.refinenet import init_refiner_model
from CRAFT.imgproc import resize_aspect_ratio, normalizeMeanVariance
import cv2
from dataset import load_npz_mmap, load_h5_dataset
import argparse
from PIL import Image
import argparse
import multiprocessing
import torch
from torch.autograd import Variable
from CRAFT.craft_utils import adjustResultCoordinates, getDetBoxes
import cProfile
import pstats
import h5py


def convert_polygons_to_boxes(polygons):
    """Convert polygons to bounding boxes"""
    boxes = []
    try:
        for poly in polygons:
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    except Exception as e:
        print(f"Error converting polygons to boxes: {e}")
    return boxes


def preprocess_image(image: np.ndarray, canvas_size: int, mag_ratio: bool):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    return x, ratio_w, ratio_h

# TODO either get preprocess running in parallel or implement this in the model
def preprocess_image_np(image: np.ndarray, canvas_size: int, mag_ratio: bool):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = normalizeMeanVariance(img_resized)
    x = np.transpose(x, (2, 0, 1))               # [h, w, c] to [c, h, w]
    return x, ratio_w, ratio_h

def resize_single_image(image, canvas_size, mag_ratio):
    """Resize a single image to the specified canvas size"""
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio
    return img_resized, ratio_w, ratio_h


def batch_preprocess_image_np(batch_images, canvas_size, mag_ratio):
    """Process a batch of images with vectorized operations where possible"""
    batch_size = len(batch_images)
    resized_images = []
    ratios_w = []
    ratios_h = []
    
    # TODO use multiprocessing here 
    for i in range(batch_size):
        img_resized, target_ratio, _ = resize_aspect_ratio(
            batch_images[i], canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio
        
        resized_images.append(img_resized)
        ratios_w.append(ratio_w)
        ratios_h.append(ratio_h)

    # with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    #     results = pool.starmap(
    #         resize_single_image,
    #         [(batch_images[i], canvas_size, mag_ratio) for i in range(batch_size)]
    #     )
    # resized_images, ratios_w, ratios_h = zip(*results)
    
    # Convert resized images into a single NumPy array
    batch_resized = np.stack(resized_images, axis=0)
    
    # Vectorized normalization (much faster than processing one by one)
    batch_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    batch_std = np.array([0.229, 0.224, 0.225] , dtype=np.float32).reshape(1, 1, 1, 3)
    
    batch_normalized = (batch_resized / 255.0 - batch_mean) / batch_std
    
    # Transpose from [B, H, W, C] to [B, C, H, W]
    batch_transposed = np.transpose(batch_normalized, (0, 3, 1, 2))
    
    return batch_transposed, ratios_w, ratios_h

def convert_polygons_to_boxes_parallel(batch_polys, num_workers=None):
    """Convert polygons to bounding boxes in parallel"""
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(batch_polys))
    
    with multiprocessing.Pool(num_workers) as pool:
        batch_boxes = pool.map(convert_polygons_to_boxes, batch_polys)
    
    return batch_boxes


def parallel_post_process(text_scores, link_scores, ratios_w, ratios_h, params):
    """Run post-processing in parallel for multiple images"""
    text_threshold, link_threshold, low_text = params
    
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(text_scores))) as pool:
        results = pool.starmap(
            process_single_image,
            [(text_scores[i], link_scores[i], ratios_w[i], ratios_h[i], 
              text_threshold, link_threshold, low_text) for i in range(len(text_scores))]
        )
    return results

def process_single_image(text_score, link_score, ratio_w, ratio_h, text_threshold, link_threshold, low_text):
    """Process a single image's score maps"""
    boxes, polys = getDetBoxes(
        text_score, link_score,
        text_threshold, link_threshold,
        low_text, False
    )
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return convert_polygons_to_boxes(boxes)


def preprocess_craft(data_dir, device="cuda", batch_size=32, resume=True, num_workers=None):
    """Preprocess CRAFT results for entire dataset"""
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), batch_size)
    
    for mode in ['train', 'test']:
        print(f"Processing {mode} set...")
        # Initialize CRAFT model - only once, outside the loop
        craft_model = CRAFTModel(
                cache_dir='weights/',
                device=device,
                use_refiner=False,
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
        
        output_file = os.path.join(data_dir, f'{mode}_craft_boxes.h5')
        with h5py.File(output_file, 'w') as h5f:  # Truncate existing file
            h5f.create_group('boxes')
            h5f['boxes'].attrs['preprocessing_batch_size'] = batch_size

        
        # Check for partial file and resume if requested        
        if resume:
            existing_batches = [f for f in os.listdir(data_dir) if f.startswith(f'batch_') and f.endswith('.npz')]
            if existing_batches:
                # Extract batch numbers and find the maximum
                batch_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_batches]
                max_batch = max(batch_numbers)
                start_idx = (max_batch + 1) * batch_size
                print(f"Resuming from batch {max_batch + 1}: {start_idx} images already processed")
            else:
                start_idx = 0
        else:
            # Remove any existing batch files to start fresh
            existing_batches = [f for f in os.listdir(data_dir) if f.startswith(f'batch_') and f.endswith('.npz')]
            for batch_file in existing_batches:
                os.remove(os.path.join(data_dir, batch_file))
            start_idx = 0

        num_images = len(images)
        for i in tqdm(range(start_idx, num_images, batch_size)):
            end_idx = min(i + batch_size, num_images)
            batch_images = images[i:end_idx][:]

            batch_tensors, ratios_w, ratios_h = batch_preprocess_image_np(
                batch_images, args.canvas_size, args.mag_ratio
            )

            batch_img_tensors = torch.from_numpy(batch_tensors).pin_memory()
            ratios_w_tensor = torch.tensor(ratios_w, device=device)
            ratios_h_tensor = torch.tensor(ratios_h, device=device)
        
            batch_polys = craft_model.get_batch_polygons(batch_img_tensors, 
                ratios_w_tensor, ratios_h_tensor
            )
            
            # Convert to torch tensor
            # NOTE that switching from convert_polygons_to_boxes to convert_polygons_to_boxes_parallel doubles the time!
            batch_boxes = [convert_polygons_to_boxes(polygons) for polygons in batch_polys]
            #batch_boxes = convert_polygons_to_boxes_parallel(batch_polys, num_workers)    
            
            output_file = os.path.join(data_dir, f'{mode}_craft_boxes.h5')
            with h5py.File(output_file, 'a') as h5f:  
                group = h5f.require_group('boxes')
                # Store each image's boxes individually
                for j, boxes in enumerate(batch_boxes):
                    img_idx = i + j
                   
                    if img_idx < num_images:  # Make sure we don't go beyond the dataset size
                        try:
                            # Ensure boxes has shape (n, 4)
                            boxes_array = np.array(boxes, dtype=np.int32)
                            if len(boxes) == 0:
                                boxes_array = np.empty((0, 4), dtype=np.int32)
                            elif boxes_array.ndim == 1:
                                # If somehow we got a 1D array, reshape it
                                boxes_array = boxes_array.reshape(-1, 4)
                            dset = group.create_dataset(
                                name=str(img_idx),
                                data=boxes_array,
                                compression="gzip"
                            )
                        except Exception as e:
                            print(f"Error storing boxes for image {img_idx}: {e}")
            # batch_file = os.path.join(data_dir, f'batch_{i // batch_size}.npz')
            # np.savez_compressed(batch_file, boxes=np.array(batch_boxes, dtype=object))
            # print(f"Saved batch {i // batch_size} to {batch_file}")
        
        # print(f"Saved {len(all_boxes)} box sets to {output_file}")
        #combine_batch_files(data_dir, mode, cleanup=True)  # Set cleanup=True if you want to delete batch files

        
        # Close H5 file if opened
        if using_h5 and h5_file_handle is not None:
            h5_file_handle.close()

def load_boxes_from_batches(list_file):
    """
    Generator function to load boxes from batch files one by one,
    without loading all into memory at once.
    
    Parameters:
    - list_file: Path to file containing list of batch files
    
    Yields:
    - boxes from each batch file
    """
    with open(list_file, 'r') as f:
        batch_files = [line.strip() for line in f]
    
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            batch_data = np.load(batch_file, allow_pickle=True)
            yield batch_data['boxes']

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Preprocess CRAFT results for font dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--no_resume", action="store_true", help="Don't resume from partial files")
    parser.add_argument("--canvas_size", type=int, default=1280, help="Canvas size for CRAFT model")
    parser.add_argument("--mag_ratio", type=float, default=1.5, help="Magnification ratio for CRAFT model")
    args = parser.parse_args()

    # profiler = cProfile.Profile()
    # profiler.enable()

    preprocess_craft(args.data_dir, device="cuda", batch_size=args.batch_size, resume=not args.no_resume)
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative').print_stats(40)  # Show top 40 functions by cumulative time
    