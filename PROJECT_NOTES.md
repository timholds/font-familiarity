
## Experiments
[ ] Try contrastive loss
[ ] Try triplet loss
[ ] Experiment with weight decay
[ ] Test single conv layers vs back to back conv layers between downsamples
[ ] Add a wandb config file and sweep conv sizes
[ ] Compare my model to a LoRA of gemmapali or a multimodal llama





[ ] move the class embedding generation into prep train test data.py? Otherwise go back to using a class embedding argument in the frontend_app.py instead of searching for it in the dataset folder 
[ ] fix the error when uploading an image
[ ] add some processing to resize images coming in and make them single channel
[ ] remove the model saving part for the sweep. once we know which model trains best we can retrain it and reanble the saving.
[ ] cleanup the metrics 
- [ ] is the epoch time metric actually telling me how long an epoch takes?


[X] Figure out how to keep track of the model experiments - use weights and biases 
[X] Test my model on cifar dataset instead to sanity check that it is capable of learning anything  
[X] create a small test dataset for iterating on the model - font-dataset-npz_test  
[X] make it so that it only saves the class averages at the end of training instead of everytime a new best model gets saved  
[X] Establish simple conv baseline for the encoder
[X] add a LR warmup and cosine learning rate  
[ ] Test a resnet as the encoder
[ ] Test a resnet with supervised contrastive loss (does it need to be a pretrain)


# Training on Alienware
- ssh timholds@192.168.68.125
    - setup ssh keys 
    - Creating a simple shell script to automate the sync + train process
- pull github repo with the latest model file 
- use github DVC w/ free tier of S3?


# Deployment
[ ] get a digital ocean ubuntu 22.04 server, 1GB RAM
- `ssh -i ~/.ssh/digitalocean root@137.184.232.187`

[ ] figure out which files and requirements I need to have setup
[ ] create a setup script to launch 
[ ] get a domain
[ ] setup a github actions to deploy to the server on push to main
    - .github/workflows/deploy.yml
## TODO 
[ ] figure out a workflow for deployment 
    - need to copy over the 
    - gather model file, embeddings file, 
    - what files do i need to copy over for the deployment? just the frontend_app - maybe make a frontend_requirements.txt? need flask, pytorch (pytroch lite?)
[ ] add test_e2e.py to github actions when merging develop into main
- update the script to automatically upload an image to the server and check the response



## ML Steps
Simple baseline model
Get a model to overfit
Regularize
Try a more complex model
Try a more complex model with more data
Try a more complex model with more data and more regularization
Try a more complex model with more data and more regularization and more data augmentation
[ ] Visualize model predictions some small holdout batch every epoch


## Bugs
[x] critical:  there's an off by one error in the labels mapping. When I try to use a super overfit model on the frontend, it returns the wrong class with super high certainty! Checking this out closer, I notice it's predicting the class 1 off from the correct class. For example, on an image of the "yesseva" font, the model predicts "yesteryear" with 99.5% certainty. 


# ML TODO
[ ] model experimentation: data augmentation - vary positioning/layout, font size and style, color and background color, text itself  
[ ] try a clip model of same font different font?  
[ ] train a classifier model and use the average class features to find which classes are closer or more similar to each other and return the top 5  
[ ] do we get anything out of top eigenvectors of the data covariance matrix  
[ ] distance between their mean images 
[ X ] how can i make the model name legible / get returned from the train script? the filename itself has some of the hyperparams baked in - for example `fontCNN_BS64-ED512-IC16.pt`  
- solution: create a ml.utils file with get_model_path() a