#!/usr/bin/env python3
import subprocess
import os
import sys
import time
import argparse

def run_command(command, description):
    """Run a command and check for errors"""
    print(f"\n=== Starting: {description} ===")
    start_time = time.time()
    
    try:
        # Run the command and display output in real-time
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for simpler handling
            text=True,
            bufsize=1
        )
        
        # Print stdout and stderr in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
        process.stdout.close()
        return_code = process.wait()
        
        # If there was an error, exit
        if return_code != 0:
            print(f"Error: {description} failed with code {return_code}")
            sys.exit(return_code)
        
        end_time = time.time()
        print(f"\n=== Completed: {description} in {end_time - start_time:.2f} seconds ===")
        return return_code
        
    except Exception as e:
        print(f"Exception occurred while running {description}: {e}")
        sys.exit(1)

def check_directory(directory, description, create=False):
    """Check if a directory exists, optionally create it"""
    if not os.path.exists(directory):
        if create:
            print(f"Creating {description} directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"Error: {description} directory does not exist: {directory}")
            sys.exit(1)
    else:
        print(f"Confirmed: {description} directory exists: {directory}")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run data processing pipeline")
    parser.add_argument("--skip_font_images", action="store_true", help="Skip creating font PIL images")
    parser.add_argument("--skip_train_test", action="store_true", help="Skip preparing train/test data")
    parser.add_argument("--skip_craft", action="store_true", help="Skip running CRAFT pre-extraction")
    
    # Input parameters
    parser.add_argument("--text_file", default="data_generation/lorem_ipsum_zipf.txt", help="Text file for font images")
    parser.add_argument("--backgrounds_dir", default="backgrounds", help="Backgrounds directory")
    parser.add_argument("--font_file", default="data_generation/fonts_test.txt", help="Font file")
    parser.add_argument("--image_resolution", default=384, type=int, help="Image resolution")
    parser.add_argument("--samples_per_class", default=10, type=int, help="Samples per class")
    parser.add_argument("--test_size", default=0.1, type=float, help="Test size ratio")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for CRAFT pre-extraction")
    
    # Output parameters
    parser.add_argument("--font_images_dir", default="test-data/font-images-384-spc10", help="Output directory for font images")
    parser.add_argument("--dataset_dir", default="test-data/dataset-384-spc10", help="Output directory for dataset")
    
    return parser.parse_args()

def main():
    args = parse_args()
    print("Starting data processing pipeline...")
    
    # Check if input directories exist
    check_directory("data_generation", "Data generation scripts")
    check_directory("ml", "ML scripts")
    check_directory(args.backgrounds_dir, "Backgrounds")
    
    # Create output directories if they don't exist
    check_directory(os.path.dirname(args.font_images_dir), "Font images parent", create=True)
    check_directory(args.font_images_dir, "Font images output", create=True)
    check_directory(os.path.dirname(args.dataset_dir), "Dataset parent", create=True)
    check_directory(args.dataset_dir, "Dataset output", create=True)
    
    # Check if input files exist
    if not os.path.exists(args.text_file):
        print(f"Error: Text file does not exist: {args.text_file}")
        sys.exit(1)
    if not os.path.exists(args.font_file):
        print(f"Error: Font file does not exist: {args.font_file}")
        sys.exit(1)
    
    # Command 1: Create font PIL images
    if not args.skip_font_images:
        cmd1 = f"python data_generation/create_font_pil_images.py --text_file {args.text_file} --backgrounds_dir {args.backgrounds_dir} --background_probability .25 --font_file {args.font_file} --image_resolution {args.image_resolution} --output_dir {args.font_images_dir} --samples_per_class {args.samples_per_class}"
        run_command(cmd1, "Creating font PIL images")
    else:
        print("Skipping: Creating font PIL images")
    
    # Command 2: Prepare train/test data
    if not args.skip_train_test:
        cmd2 = f"python data_generation/prep_train_test_data.py --input_image_dir {args.font_images_dir}/ --output_dir {args.dataset_dir} --test_size {args.test_size}"
        run_command(cmd2, "Preparing train/test data")
    else:
        print("Skipping: Preparing train/test data")
    
    # Command 3: Run CRAFT pre-extraction
    if not args.skip_craft:
        # Use dataset_dir instead of font_images_dir
        cmd3 = f"python ml/craft_preextract.py --data_dir {args.dataset_dir}/ --batch_size {args.batch_size}"
        run_command(cmd3, "Running CRAFT pre-extraction")
    else:
        print("Skipping: Running CRAFT pre-extraction")
    
    print("\n🎉 All processing completed successfully! 🎉")

if __name__ == "__main__":
    main()