import pytest
import os
from pathlib import Path
import time
import logging
import subprocess
import json
import torch
import numpy as np
from PIL import Image
import requests
from typing import Optional
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineValidationError(Exception):
    """Custom exception for pipeline validation failures"""
    pass

def validate_image_generation(image_dir: Path) -> None:
    """Validate that font images were generated correctly"""
    if not image_dir.exists():
        raise PipelineValidationError(f"Image directory {image_dir} does not exist")
    
    # Check if directory has font subdirectories
    font_dirs = [d for d in image_dir.iterdir() if d.is_dir()]
    if not font_dirs:
        raise PipelineValidationError(f"No font directories found in {image_dir}")
    
    # Check image properties of first image in each font dir
    for font_dir in font_dirs:
        images = list(font_dir.glob("*.jpg"))
        if not images:
            raise PipelineValidationError(f"No images found in {font_dir}")
        
        # Check first image properties
        sample_image = Image.open(images[0])
        if sample_image.mode != 'L':  # Should be grayscale
            raise PipelineValidationError(f"Image {images[0]} is not grayscale")
        if sample_image.size != (128, 128):  # Check resolution
            raise PipelineValidationError(
                f"Image {images[0]} has wrong resolution: {sample_image.size}"
            )

def validate_dataset_prep(dataset_dir: Path) -> None:
    """Validate that dataset was prepared correctly"""
    required_files = ['train.npz', 'test.npz', 'label_mapping.npy']
    for file in required_files:
        if not (dataset_dir / file).exists():
            raise PipelineValidationError(f"Required file {file} not found in {dataset_dir}")
    
    # Load and validate train/test data
    train_data = np.load(dataset_dir / 'train.npz')
    test_data = np.load(dataset_dir / 'test.npz')
    
    # Check data structure
    for data in [train_data, test_data]:
        if 'images' not in data or 'labels' not in data:
            raise PipelineValidationError("Dataset files missing required arrays")
        
        # Validate shapes match
        if len(data['images']) != len(data['labels']):
            raise PipelineValidationError("Mismatch between images and labels lengths")

def validate_model(model_path: Path, expected_params: dict) -> None:
    """Validate the trained model file and its parameters"""
    if not model_path.exists():
        raise PipelineValidationError(f"Model file {model_path} does not exist")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check required keys in checkpoint
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        for key in required_keys:
            if key not in checkpoint:
                raise PipelineValidationError(f"Model checkpoint missing required key: {key}")
        
        # Verify model completed expected number of epochs
        if checkpoint['epoch'] < expected_params['epochs']:
            raise PipelineValidationError(
                f"Model training incomplete. Expected {expected_params['epochs']} epochs, "
                f"got {checkpoint['epoch']}"
            )
        
        # Verify model architecture matches expected parameters
        state_dict = checkpoint['model_state_dict']
        embedding_layer = state_dict.get('embedding_layer.0.weight')
        if embedding_layer is None:
            raise PipelineValidationError("Model missing embedding layer")
        
        if embedding_layer.shape[0] != expected_params['embedding_dim']:
            raise PipelineValidationError(
                f"Wrong embedding dimension. Expected {expected_params['embedding_dim']}, "
                f"got {embedding_layer.shape[0]}"
            )
            
    except Exception as e:
        raise PipelineValidationError(f"Error validating model: {e}")

def validate_embeddings(embeddings_path: Path, model_path: Path) -> None:
    """Validate the generated embeddings file"""
    if not embeddings_path.exists():
        raise PipelineValidationError(f"Embeddings file {embeddings_path} does not exist")
    
    try:
        # Load embeddings and model
        embeddings = np.load(embeddings_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get number of classes from model
        classifier_weight = checkpoint['model_state_dict']['classifier.weight']
        num_classes = classifier_weight.shape[0]
        
        # Validate embeddings shape
        if embeddings.shape[0] != num_classes:
            raise PipelineValidationError(
                f"Embeddings have wrong number of classes. Expected {num_classes}, "
                f"got {embeddings.shape[0]}"
            )
            
        # Validate embedding dimension matches model
        embedding_dim = checkpoint['model_state_dict']['embedding_layer.0.weight'].shape[0]
        if embeddings.shape[1] != embedding_dim:
            raise PipelineValidationError(
                f"Wrong embedding dimension. Expected {embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
            
    except Exception as e:
        raise PipelineValidationError(f"Error validating embeddings: {e}")

def validate_frontend(port: int) -> None:
    """Validate that the frontend server is responding"""
    url = f"http://localhost:{port}"
    max_retries = 5
    retry_delay = 2
    
    for i in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.info("Frontend server is responding")
            return
        except requests.RequestException:
            if i < max_retries - 1:
                logger.info(f"Frontend not ready, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise PipelineValidationError("Frontend server failed to respond")

def run_step(command: str, description: str) -> Optional[str]:
    """Run a pipeline step and capture output"""
    logger.info(f"Starting: {description}")
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Completed: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Step failed: {description}")
        logger.error(f"stdout: {e.stdout}")  # Add this line
        logger.error(f"stderr: {e.stderr}")  # Add this line
        raise PipelineValidationError(f"Pipeline step failed: {e}")

@contextmanager
def frontend_server(command: str, port: int):
    """Start and manage the frontend server process"""
    process = subprocess.Popen(command.split())
    try:
        # Wait for server to start
        time.sleep(5)
        validate_frontend(port)
        yield process
    finally:
        # Keep server running for manual testing
        logger.info("Frontend server is running. Press Ctrl+C to stop.")
        try:
            process.wait(timeout=300)  # Wait for 5 minutes
        except subprocess.TimeoutExpired:
            logger.info("Shutting down frontend server")
            process.terminate()
            process.wait()

def test_full_pipeline():
    """Run and validate the complete pipeline"""
    try:
        # Step 1: Generate font images
        image_dir = Path("data/font-images-test")
        run_step(
            "python data_generation/create_font_images.py "
            "--text_file data_generation/lorem_ipsum.txt "
            "--font_file data_generation/fonts_test.txt "
            f"--output_dir {image_dir} "
            "--samples_per_class 10 "
            "--image_resolution 128 "
            "--port 5100 "
            "--font_size 35 "
            "--line_height 1.5",
            "Font image generation"
        )
        validate_image_generation(image_dir)

        # Step 2: Prepare dataset
        dataset_dir = Path("data/font_dataset_npz_test")
        run_step(
            "python data_generation/prep_train_test_data.py "
            f"--input_image_dir {image_dir} "
            f"--output_dir {dataset_dir} "
            "--test_size .1",
            "Dataset preparation"
        )
        validate_dataset_prep(dataset_dir)

        # Step 3: Train model
        expected_params = {
            'epochs': 1,
            'batch_size': 64,
            'learning_rate': .0001,
            'weight_decay': .01,
            'embedding_dim': 128,
            'resolution': 64,
            'initial_channels': 16,
        }
        
        run_step(
            "python ml/train.py "
            f"--data_dir {dataset_dir} "
            f"--epochs {expected_params['epochs']} "
            f"--batch_size {expected_params['batch_size']} "
            f"--learning_rate {expected_params['learning_rate']} "
            f"--weight_decay {expected_params['weight_decay']} "
            f"--embedding_dim {expected_params['embedding_dim']} "
            f"--resolution {expected_params['resolution']} "
            f"--initial_channels {expected_params['initial_channels']}",
            "Model training"
        )

        model_path = Path(f"fontCNN_BS{expected_params['batch_size']}-"
                         f"ED{expected_params['embedding_dim']}-"
                         f"IC{expected_params['initial_channels']}.pt")
        validate_model(model_path, expected_params)

        # Step 4: Create embeddings
        embeddings_file = "class_embeddings.npy"
        embeddings_path = os.path.join(dataset_dir, embeddings_file)
        run_step(
            "python create_embeddings.py "
            f"--model_path {model_path} "
            f"--data_dir {dataset_dir} "
            f"--embeddings_file {embeddings_path}",
            "Creating embeddings"
        )
        validate_embeddings(embeddings_path, model_path)

        # Step 5: Start frontend server
        frontend_port = 8080
        with frontend_server(
            f"python frontend_app.py "
            f"--model_path {model_path} "
            f"--data_dir {dataset_dir} "
            f"--embedding_file {embeddings_path} "
            f"--port {frontend_port}",
            frontend_port
        ):
            logger.info("Pipeline completed successfully!")
            
    except PipelineValidationError as e:
        logger.error(f"Pipeline validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def test_image_generation(image_dir):
    run_step(
        "python data_generation/create_font_images.py "
        "--text_file data_generation/lorem_ipsum.txt "
        "--font_file data_generation/fonts_test.txt "
        f"--output_dir {image_dir} "
        "--samples_per_class 10 "
        "--image_resolution 128 "
        "--port 5100 "
        "--font_size 35 "
        "--line_height 1.5",
        "Font image generation"
    )
    validate_image_generation(image_dir)

def test_dataset_prep(image_dir, dataset_dir):
    run_step(
        "python data_generation/prep_train_test_data.py "
        f"--input_image_dir {image_dir} "
        f"--output_dir {dataset_dir} "
        "--test_size .1",
        "Dataset preparation"
    )
    validate_dataset_prep(dataset_dir)

def test_model_training(dataset_dir, model_path, expected_params):
    
    run_step(
        "python ml/train.py "
        f"--data_dir {dataset_dir} "
        f"--epochs {expected_params['epochs']} "
        f"--batch_size {expected_params['batch_size']} "
        f"--learning_rate {expected_params['learning_rate']} "
        f"--weight_decay {expected_params['weight_decay']} "
        f"--embedding_dim {expected_params['embedding_dim']} "
        f"--resolution {expected_params['resolution']} "
        f"--initial_channels {expected_params['initial_channels']}",
        "Model training"
    )

    
    validate_model(model_path, expected_params)
    return model_path

def test_create_embeddings(dataset_dir, model_path, embeddings_path):
    # dataset_dir = Path("data/font_dataset_npz_test")
    # model_path = Path("fontCNN_BS64-ED128-IC16.pt")
    # TODO 
    embeddings_file = "class_embeddings.npy"
    embeddings_path = os.path.join(dataset_dir, embeddings_file)
    run_step(
        "python create_embeddings.py "
        f"--model_path {model_path} "
        f"--data_dir {dataset_dir} "
        f"--embeddings_file {embeddings_path}",
        "Creating embeddings"
    )
    validate_embeddings(embeddings_path, model_path)

def test_frontend_server(model_path, dataset_dir, embeddings_path):
    frontend_port = 8080
    with frontend_server(
        f"python frontend_app.py "
        f"--model_path {model_path} "
        f"--data_dir {dataset_dir} "
        f"--embedding_file {embeddings_path} "
        f"--port {frontend_port}",
        frontend_port
    ):
        logger.info("Pipeline completed successfully!")

def run_unit_tests():
    expected_params = {
        'epochs': 1,
        'batch_size': 64,
        'learning_rate': .0001,
        'weight_decay': .01,
        'embedding_dim': 128,
        'resolution': 64,
        'initial_channels': 16,
    }
    image_dir = Path("data/font-images-test")
    dataset_dir = Path("data/font_dataset_npz_test")
    test_image_generation(image_dir)
    # test_dataset_prep(image_dir, dataset_dir)
    # # model_file = f"fontCNN_BS{expected_params['batch_size']}-"
    # #                 f"ED{expected_params['embedding_dim']}-"
    # #                 f"IC{expected_params['initial_channels']}.pt"
    # model_file = f"fontCNN_BS{expected_params['batch_size']}\
    #     -ED{expected_params['embedding_dim']}\
    #     -IC{expected_params['initial_channels']}.pt"
    # model_path = os.path.join(dataset_dir, model_file)
    # test_model_training(dataset_dir, model_path)
    # embeddings_path = os.path.join(dataset_dir, "class_embeddings_test.npy")
    # test_create_embeddings(dataset_dir, model_path, embeddings_path)
    # test_frontend_server(model_path, dataset_dir, embeddings_path)

if __name__ == "__main__":
    run_unit_tests()
    #test_full_pipeline()