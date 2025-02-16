# Quickstart
Create a new virtual environment and install the requirements. I'm using Python 3.10.12, use other versions and YMMV
```
python -m venv font-env
source font-env/bin/activate
pip install -r requirements.txt
```


To run any of the individual components from the root of font-familiarity:
- `python data_generation/create_font_images.py --text_file data_generation/lorem_ipsum.txt --font_file data_generation/fonts_test.txt --output_dir test-data/font-images --samples_per_class 10 --image_resolution 128 --port 5100 --font_size 35 --line_height 1.5`  
- `python data_generation/prep_train_test_data.py --input_image_dir test-data/font-images --output_dir test-data/font-dataset-npz --test_size .1`  
- `python ml/train.py --data_dir "test-data/font-dataset-npz" --epochs 10 --batch_size 64 --learning_rate .001 --weight_decay .01 --embedding_dim 256 --resolution 64 --initial_channels 16`  
- `python create_embeddings.py --model_path test-data/font-dataset-npz/fontCNN_BS64-ED256-IC16.pt --data_dir test-data/font-dataset-npz --embeddings_file class_embeddings_256.npy`  
- `python frontend_app.py --model_path test-data/font-dataset-npz/fontCNN_BS64-ED256-IC16.pt --data_dir test-data/font-dataset-npz --embeddings_path test-data/font-dataset-npz/class_embeddings_256.npy --port 8080`  


To run all of these commands in sequence: `python test_e2e.py` 

It creates a tiny dataset, preprocesses the dataset, trains a model for an epoch, creates class embeddings, and runs the frontend. If running correctly, it should create a new folder `test-data` with images images and train/test data, as well as saving a model and embeddings. Finally, you should be go to the frontend at `localhost:8080`, upload one of the test images, and verify there are no errors. Note that we aren't expecting the model to actually perform well, but everything else should be working.

# Overview  

**Problem Statement**: If you see a font in the wild and want to use it yourself, how do you know the name of the font? If its a paid font, how do you find similar free fonts? The goal of this project is to build a tool that can take an image of a font and return the most similar free fonts from Google's ~700 fonts. We will do this in three steps: 1) generating the dataset of different fonts, 2) training a model to learn the features of each font, and 3) building a frontend to take in an image with some text and return the most similar fonts.

## Dataset
Best to visit the data generation readme [Dataset Generation Readme](data_generation/README.md)

A super rough, ~15GB font dataset is available to download on Huggingface https://huggingface.co/datasets/Timholds/Fonts/tree/main. The images are 256x256 grayscale jpg files with no data augmentation applied. 


## Model and embeddings
Our goal is to find which fonts are most similar to the unknown input font, so we need to have some idea of what all the *known* fonts look like in feature space and return the closest ones. To find that, we can take the trained model and for each class, average the output embedding over all the training examples to give us a prototype average for each class. 

For each font class (out of the ~700 fonts), we're:
- Taking all images of that font (1000 images per class in this case)
- Running each through the model to get its 1024-dimensional feature vector
- Computing the average of all these vectors for that class

So if Font_A with n images, we get one 1024-dim vector representing the "average characteristics" of Font_A. When a user inputs their image, we run it through the trained model and use the activations as a query to find the closest font embeddings. 

## Frontend


## Main Software Pieces
- dataset generation
    - html template to render the text into fonts and create an html page
    - flask server to put the html onto a browser 
    - selenium scraper to takes screenshots of those webpages
    - script to orchestrate this and save the data to disk as npz or pkl files
- model training and font embedding creation
    - model architecture file
    - dataset file to load dataset into pytorch DataLoader
    - training file to handle the training loop, evaluation, metrics
    - create class-average font embeddings using the trained model and dataset
- model api and frontend
    - interface to take upload image, run it through the embedding model, do cosine similarity against and return the most similar fonts  

<!-- <br>  </br>   -->

# Data generation
**Note**: The data generation step is the most time consuming and can take a few hours to generate a few thousand images.  
**Note**: `output_dir` should be the same between these scripts to make sure you are preparing the data you just generated.  
### Generate the images 
`python data_generation/create_font_images.py --text_file data_generation/lorem_ipsum.txt --font_file data_generation/full_fonts_list.txt --output_dir data/font-images --samples_per_class 100 --image_resolution 128 --port 5100 --font_size 35 --line_height 1.5`  
### Process the images into Cifar like train/test format
`python data_generation/prep_train_test_data.py --input_image_dir data/font-images --output_dir data/font-dataset-npz --test_size .1`

# Train Model and Generate Embeddings
**Note**: `output_dir` from the `prep_train_test_data.py` should be the same as the `data_dir` in the `train.py` and `create_embeddings.py` script.
### Train the model
`python ml/train.py --data_dir data/font-dataset-npz --epochs 30 --batch_size 64 --learning_rate 0.0001 --weight_decay 0.01 --embedding_dim 128 --resolution 64 --initial_channels 16`

### Create embeddings with the trained model
`python create_embeddings.py --model_path fontCNN_BS64-ED512-IC16.pt --data_dir data/font-dataset-npz -- embeddings_file class_embeddings_512.npy --output_path data/font-dataset-npz` `


# Frontend
**Note**: `data_dir` from the `train.py` and `create_embeddings.py` should be the same as the `data_dir` in the `frontend_app.py` script.
**Note**: The model pt passed in needs to have an embedding dimension that matches the class embeddings. The model file has the embedding dimension in the name. For example, fontCNN_BS64-ED512-IC16.pt has an embedding dimension of 512.
`python frontend_app.py --model_path fontCNN_BS64-ED512-IC16.pt --data_dir data/font-dataset-npz --embedding_file class_embeddings.npy --port 8080`



# Example file structure
- test-data
    - font-images
        - abeeze
            - image0000.jpg
            - ...
            - image0010.jpg
        - ...
        - archivo narrow
            - image0000.jpg
            - ...
            - image0010.jpg
    - font-images-npz
        - label_mapping.npy ~1kb
        - train.npz - few MB
        - test.npz ~500kb
        - fontCNN_BS64-ED128-IC16.pt ~5mb
        - class_embeddings.test ~25kb



# Similar tools
The first 3 google results for font finders failed to recognize any of the font images I uploaded. These were high quality 256x256 images, much better quality than I'm training on. They all seem to segment the characters individually and then match against the characters. 

I also tried a "Font Identifier AI" in the ChatGPT store that failed to recognize any of the fonts images I tried.  

The Font Finder Chrome plugin seems quite promising since it uses the information from the browser's elements / CSS. However the format it returns in isn't all that helpful: font-family (stack)

![Font Plugin Example](font-plugin.png)
https://chromewebstore.google.com/detail/font-finder/bhiichidigehdgphoambhjbekalahgha

