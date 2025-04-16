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

# from CRAFT import imgproc
link_threshold = 1.9
text_threshold = .5
low_text = .5


HF_MODELS = {
    'craft': dict(
        repo_id='boomb0om/CRAFT-text-detector',
        filename='craft_mlt_25k.pth',
    ),
    'refiner': dict(
        repo_id='boomb0om/CRAFT-text-detector',
        filename='craft_refiner_CTW1500.pth',
    )
}

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

def batch_preprocess_image_np(batch_images, canvas_size, mag_ratio):
    """Process a batch of images with vectorized operations where possible"""
    batch_size = len(batch_images)
    resized_images = []
    ratios_w = []
    ratios_h = []
    
    # Process each image for resizing (can't be easily vectorized due to aspect ratio preservation)
    for i in range(batch_size):
        img_resized, target_ratio, _ = resize_aspect_ratio(
            batch_images[i], canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio
        
        resized_images.append(img_resized)
        ratios_w.append(ratio_w)
        ratios_h.append(ratio_h)
    
    # Stack resized images for batch normalization
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

        # paths = {"craft": os.path.join(os.getcwd(), "weights/models--boomb0om--CRAFT-text-detector/snapshots/3b6fb468e75c3cf833875e2b073e7ea3c477975a/craft_mlt_25k.pth")}
        # paths["refiner"] = os.path.join(os.getcwd(), "weights/models--boomb0om--CRAFT-text-detector/snapshots/3b6fb468e75c3cf833875e2b073e7ea3c477975a/craft_refiner_CTW1500.pth")
        # craft_net = init_CRAFT_model(paths['craft'], "cuda", fp16=True)
        # refiner_net = init_refiner_model(paths['refiner'], "cuda")
        
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
        partial_file = output_file + ".partial.npz"
        
        all_boxes = []
        # Check for partial file and resume if requested        
        if resume and os.path.exists(partial_file):
            try:
                partial_data = np.load(partial_file, allow_pickle=True)
                all_boxes = list(partial_data['boxes'])
                start_idx = len(all_boxes)
                print(f"Resuming from checkpoint: {start_idx} images already processed")
            except Exception as e:
                print(f"Error loading partial file: {e}")
                print("Starting from the beginning")
                start_idx = 0
        else:
            start_idx = 0

        num_images = len(images)
        
        # run the batch of images through craft and do the post processing in parallel
        
        for i in tqdm(range(start_idx, num_images, batch_size)):
            # batch_images = [Image.fromarray(images[j].astype(np.uint8)) for j in batch_indices]
            end_idx = min(i + batch_size, num_images)
            batch_images = images[i:end_idx][:]

            batch_tensors, ratios_w, ratios_h = batch_preprocess_image_np(
                batch_images, args.canvas_size, args.mag_ratio
            )
            batch_img_tensors = torch.from_numpy(batch_tensors)
            
            # Preprocess images
            # TODO parallelize this later and just stack tensors for now()
            # need to just get the image from first item in image, ratio-h, ratio_w
            # list of tups length batch size (image array, ratio_w, ratio_h)
            # Unpack preprocessed_results into separate arrays
            # preprocessed_results = [preprocess_image_np(image, args.canvas_size, args.mag_ratio) for image in batch_images]
            # image_arrays = [result[0] for result in preprocessed_results]  # Extract the image arrays
            # ratios_w = [result[1] for result in preprocessed_results]      # Extract the ratio_w values
            # ratios_h = [result[2] for result in preprocessed_results]      # Extract the ratio_h values
            # batch_img_tensors_np = np.stack([image for image in image_arrays], axis=0)  # Stack the images into a batch tensor
            # batch_img_tensors = torch.from_numpy(batch_img_tensors_np) # BCHWC input

            # breakpoint()
            batch_polys = craft_model.get_batch_polygons(batch_img_tensors, 
                torch.tensor(ratios_w, device=device),  # Send ratios to GPU
                torch.tensor(ratios_h, device=device)
            )
            
            # Convert to torch tensor
            # NOTE that switching from convert_polygons_to_boxes to convert_polygons_to_boxes_parallel doubles the time!
            batch_boxes = [convert_polygons_to_boxes(polygons) for polygons in batch_polys]
            # batch_boxes = convert_polygons_to_boxes_parallel(batch_polys, num_workers)    
            all_boxes.extend(batch_boxes)
            # breakpoint()
            # Process each image in the batch

            # Save incrementally to avoid losing progress
            if i % (batch_size * 10) == 0:
                np.savez_compressed(
                    output_file + ".partial",
                    boxes=np.array(all_boxes, dtype=object)
                )
                
                print(f"Saved partial progress ({len(all_boxes)} image boxes) to {output_file}.partial")
            


        #     # Process batch with CRAFT - sequential approach
        #     batch_boxes = []
        #     breakpoint()
        #     for img in batch_images:
        #         try:
        #             try:
        #                 polygons = craft_model.get_polygons(img)
        #             except RuntimeError as e:
        #                 if "CUDA" in str(e) and device == "cuda":
        #                     print("CUDA error detected, falling back to CPU for this image")
        #                     # Create a temporary CPU model for fallback
        #                     cpu_model = CRAFTModel(
        #                         cache_dir='weights/',
        #                         device="cpu",
        #                         use_refiner=True,
        #                         fp16=False,  # Must be False for CPU
        #                         link_threshold=1.9,
        #                         text_threshold=.5,
        #                         low_text=.5,
        #                     )
        #                     polygons = cpu_model.get_polygons(img)
        #                 else:
        #                     # Re-raise if it's not a CUDA error
        #                     raise
                    
        #             batch_boxes = convert_polygons_to_boxes(polygons)
        #         except Exception as e:
        #             print(f"Error processing image: {e}")
        #             batch_boxes.append([])
            
        #     all_boxes.extend(batch_boxes)
        
        #     # Save incrementally to avoid losing progress
        #     if i % (batch_size * 10) == 0:
        #         np.savez_compressed(
        #             output_file + ".partial",
        #             boxes=np.array(all_boxes, dtype=object)
        #         )
                
        #         print(f"Saved partial progress ({len(all_boxes)} image boxes) to {output_file}.partial")
            
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

    
    parser = argparse.ArgumentParser(description="Preprocess CRAFT results for font dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--no_resume", action="store_true", help="Don't resume from partial files")
    parser.add_argument("--canvas_size", type=int, default=1280, help="Canvas size for CRAFT model")
    parser.add_argument("--mag_ratio", type=float, default=1.5, help="Magnification ratio for CRAFT model")
    args = parser.parse_args()

    # torch.backends.cudnn.benchmark = True
    # if torch.cuda.is_available():
    #     # Limit GPU memory to 70% of available
    #     total_mem = torch.cuda.get_device_properties(0).total_memory
    #     torch.cuda.set_per_process_memory_fraction(0.7)
    #     print(f"Limited CUDA memory to 70% of {total_mem/(1024**3):.2f} GB")


    multiprocessing.set_start_method('spawn', force=True)
    preprocess_craft(args.data_dir, device="cuda", batch_size=args.batch_size, resume=not args.no_resume)