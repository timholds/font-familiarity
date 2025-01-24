
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



# Deployment
[ ] get a digital ocean ubuntu 22.04 server, 1GB RAM
[ ] figure out which files and requirements I need to have setup
[ ] create a setup script to launch 
[ ] get a domain

## TODO 
[ ] figure out a workflow for deployment 
    - need to copy over the 
    - gather model file, embeddings file, 
    - what files do i need to copy over for the deployment? just the frontend_app - maybe make a frontend_requirements.txt? need flask, pytorch (pytroch lite?)

`ssh -i ~/.ssh/digitalocean root@137.184.232.187`


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