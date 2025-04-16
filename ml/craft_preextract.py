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
# from CRAFT import imgproc

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

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    # permute with numpy to chw
    x = np.transpose(x, (2, 0, 1))               # [h, w, c] to [c, h, w]

    # x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    # going to stack them in the batch in other function 
    # x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    return x, ratio_w, ratio_h


def preprocess_craft(data_dir, device="cuda", batch_size=32, resume=True):
    """Preprocess CRAFT results for entire dataset"""
    for mode in ['train', 'test']:
        print(f"Processing {mode} set...")

        paths = {"craft": os.path.join(os.getcwd(), "weights/models--boomb0om--CRAFT-text-detector/snapshots/3b6fb468e75c3cf833875e2b073e7ea3c477975a/craft_mlt_25k.pth")}
        paths["refiner"] = os.path.join(os.getcwd(), "weights/models--boomb0om--CRAFT-text-detector/snapshots/3b6fb468e75c3cf833875e2b073e7ea3c477975a/craft_refiner_CTW1500.pth")
        craft_net = init_CRAFT_model(paths['craft'], "cuda", fp16=True)
        refiner_net = init_refiner_model(paths['refiner'], "cuda")
        
        # Initialize CRAFT model - only once, outside the loop
        # craft_model = CRAFTModel(
        #         cache_dir='weights/',
        #         device=device,
        #         use_refiner=True,
        #         fp16=(device == "cuda"),  # Use fp16 only on CUDA
        #         link_threshold=1.9,
        #         text_threshold=.5,
        #         low_text=.5,
        #     )
        
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
        partial_file = output_file + ".partial"
        
        # Check for partial file and resume if requested
        start_idx = 0
        all_boxes = []
        
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
                all_boxes = []
        
        num_images = len(images)
        # run the batch of images through craft and do the post processing in parallel
        
        for i in tqdm(range(0, num_images, batch_size)):
            batch_indices = range(i, min(i + batch_size, num_images))
            # batch_images = [Image.fromarray(images[j].astype(np.uint8)) for j in batch_indices]
            batch_images = images[i:i+batch_size][:]

            # TODO convert images to craft format (including preprocessing)
            # maybe multiprocess calling preprocess_image()?
            # need the thing going into the model to be BCHW tensor normalized
            
            # Preprocess images
            # TODO parallelize this later and just stack tensors for now()
            breakpoint()
            preprocssed_results = [preprocess_image_np(image, args.canvas_size, args.mag_ratio) for image in batch_images]
            # need to just get the image from first item in image, ratio-h, ratio_w
            batch_img_tensors_np = np.stack([results[0] for results in preprocssed_results])
            batch_img_tensors = torch.from_numpy(batch_img_tensors_np).float()

            if device == "cuda":
                batch_img_tensors = batch_img_tensors.cuda()

            with torch.no_grad():
                breakpoint()
                y, features = craft_net.forward(batch_img_tensors)



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