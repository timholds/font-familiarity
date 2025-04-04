import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from dataset import get_dataloaders, get_char_dataloaders
from model import SimpleCNN
from char_model import CRAFTFontClassifier

import argparse
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.optim import AdamW
from metrics import ClassificationMetrics
import os
from prettytable import PrettyTable
import numpy as np
from utils import get_model_path, check_char_model_batch_independence

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



# TODO update this to work with character patches
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                warmup_epochs, warmup_scheduler, main_scheduler, metrics_calculator, char_model=False):
    """Train for one epoch, supporting both character-based and whole-image models."""
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, batch_data in enumerate(pbar):
        metrics_calculator.start_batch()
        
        if char_model:
            # For character-based model (with CRAFT)
            # Batch data is a dictionary with images, labels, and annotations
            images = batch_data['images'].to(device)
            targets = batch_data['labels'].to(device)
            annotations = batch_data['annotations'] if 'annotations' in batch_data else None
            batch_size = targets.size(0)

            # Forward pass - pass everything to the model
            optimizer.zero_grad()

            # TODO visualize model inputs here to validate range and shape

            outputs = model(images, targets, annotations)
            
            # Handle different output formats from model
            if isinstance(outputs, tuple):
                # Model returns logits and attention weights
                logits, _ = outputs
            elif isinstance(outputs, dict):
                # Model returns a dict with different outputs
                logits = outputs['logits']
            else:
                # Model directly returns logits
                logits = outputs
        else:
            data, targets = batch_data
            data, targets = data.to(device), targets.to(device)
            batch_size = targets.size(0)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(data)
        
        # Compute loss and backward
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        
        # Update schedulers
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Update running statistics
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        pred = logits.argmax(dim=1)
        total_correct += (pred == targets).sum().item()
        total_samples += batch_size
        current_acc = 100. * total_correct / total_samples
        
        if batch_idx % 100 == 0:
            # Visualize a few samples from the batch
            vis_path = f"debug/epoch_{epoch}_batch_{batch_idx}"

            # Extract patches for visualization
            if char_model and hasattr(model, 'visualize_char_preds'):
                with torch.no_grad(): 
                    patch_data = model.extract_patches_with_craft(images) # images BHWC
                    model.visualize_char_preds(
                        patches=patch_data['patches'],
                        attention_mask=patch_data['attention_mask'],
                        predictions=pred,
                        targets=targets,
                        save_path=vis_path
                    )

        
            # Add CRAFT detection visualization
            if char_model and hasattr(model, 'craft') and hasattr(model, 'visualize_craft_detections'):
                model.visualize_craft_detections(
                    images=images,  # Original images
                    targets=targets,
                    label_mapping=train_loader.dataset.label_mapping,
                    save_path=vis_path
                )

        # Compute and log batch metrics
        if batch_idx % 50 == 0:
            batch_metrics = {
                'train/batch_loss': loss.item(),
                'train/running_loss': running_loss,
                'train/running_acc': current_acc,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'global_step': batch_idx + epoch * len(train_loader)
            }
            wandb.log(batch_metrics)
        
        # Update progress bar with basic metrics
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'avg_loss': f'{running_loss:.3f}',
            'acc': f'{current_acc:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # Compute final training metrics
    train_metrics = {
        'train/loss': running_loss,
        'train/top1_acc': current_acc,
        'train/samples': total_samples,
    }
    
    return train_metrics

def evaluate(model, test_loader, criterion, device, metrics_calculator, epoch=None, char_model=False):
    """Evaluate the model, supporting both character-based and whole-image models."""
    model.eval()
    test_loss = 0
    all_logits = []
    all_targets = []
    
    # Basic accuracy tracking
    correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc='Evaluating'):
            # Handle different data formats based on model type
            if char_model:
                # For character-based model with CRAFT
                images = batch_data['images'].to(device)
                targets = batch_data['labels'].to(device)
                annotations = batch_data['annotations'] if 'annotations' in batch_data else None

                # Forward pass with whole images
                outputs = model(images, targets, annotations)

                # Extract logits from outputs
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                            
            # Accumulate tensors for advanced metrics
            all_logits.append(logits)
            all_targets.append(targets)
            
            # Compute basic metrics on the fly
            test_loss += criterion(logits, targets).item()
            
            # Top-1 accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            
            # Top-5 accuracy
            _, pred5 = logits.topk(5, 1, True, True)
            target_expanded = targets.view(-1, 1).expand_as(pred5)
            correct5 = pred5.eq(target_expanded).any(dim=1).sum().item()
            top5_correct += correct5
            
            total += targets.size(0)
    
    # Compute basic metrics
    avg_loss = test_loss / len(test_loader)
    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / total
    
    # Initialize metrics dictionary with basic metrics
    test_metrics = {
        'test/loss': avg_loss,
        'test/top1_acc': top1_acc,
        'test/top5_acc': top5_acc,
        'test/samples': total,
    }
    
    # Compute advanced metrics on the entire test set
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Get advanced metrics from the metrics calculator
    advanced_metrics = metrics_calculator.compute_all_metrics(
        logits=all_logits,
        targets=all_targets,
        model=model,
        epoch=epoch
    )
    
    # Add advanced metrics with 'test/' prefix, excluding duplicates
    for key, value in advanced_metrics.items():
        # Skip metrics we already have
        if key not in ['top1_acc', 'top5_acc', 'loss']:
            test_metrics[f'test/{key}'] = value
    
    return test_metrics


def compute_class_embeddings(model, dataloader, num_classes, device):
    """Compute and store average embeddings for each class."""
    print("\nComputing class embeddings...")
    model.eval()
    
    # Initialize storage for embeddings and counts
    class_embeddings = torch.zeros(num_classes, 1024).to(device)  # 1024 is embedding dim
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Computing embeddings'):
            data, target = data.to(device), target.to(device)
            
            # Get embeddings (features before final classification layer)
            embeddings = model.get_embedding(data)
            
            # Accumulate embeddings for each class
            for i in range(len(target)):
                class_idx = target[i].item()
                class_embeddings[class_idx] += embeddings[i]
                class_counts[class_idx] += 1
    
    # Compute averages
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_embeddings[i] /= class_counts[i]
    
    # Verify no classes were empty
    empty_classes = (class_counts == 0).sum().item()
    if empty_classes > 0:
        print(f"Warning: {empty_classes} classes had no samples!")
    
    return class_embeddings

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Checking environment:")
        print(f"CUDA_HOME environment variable: {os.environ.get('CUDA_HOME', 'Not set')}")
        print(f"CUDA_PATH environment variable: {os.environ.get('CUDA_PATH', 'Not set')}")
        print(f"LD_LIBRARY_PATH environment variable: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/font_dataset_npz_test/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--initial_channels", type=int, default=16)
    parser.add_argument("--char_model", action="store_true", help="Use character-based model")
    
    args = parser.parse_args()
    warmup_epochs = max(args.epochs // 5, 1)

    torch.manual_seed(0)
    
    # Initialize wandb
    wandb.init(
        project="Font-Familiarity",
        name=f"experiment_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            **vars(args),
            "architecture": "CNN",
            "warmup_epochs": warmup_epochs,
            "optimizer": "AdamW"
        }
    )
    
    # Set up metrics logging
    wandb.define_metric("batch_loss", step_metric="global_step")
    wandb.define_metric("batch_acc", step_metric="global_step")
    wandb.define_metric("*", step_metric="epoch")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    # Load data
    print("Loading data...")
    if args.char_model:
        train_loader, test_loader, num_classes = get_char_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
    else:
        train_loader, test_loader, num_classes = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )

    print("\nStep 2: Validating label mapping...")
    label_mapping_path = os.path.join(args.data_dir, 'label_mapping.npy')
    label_mapping = np.load(label_mapping_path, allow_pickle=True).item()
    
    assert num_classes == len(label_mapping), (
        f"Critical Error: Mismatch between dataset classes ({num_classes}) "
        f"and label mapping entries ({len(label_mapping)})"
    )
    
    # Print label mapping info
    print(f"Label mapping contains {len(label_mapping)} classes")
    print("First 5 classes:", list(label_mapping.items())[:5])
    print("Last 5 classes:", list(label_mapping.items())[-5:])

    print(f"num training batches {len(train_loader)}")
    print(f"num training datapoints {len(train_loader.dataset)}")
    print(f"num test batches {len(test_loader)}")
    print(f"num tes datapoints {len(test_loader.dataset)}")
    
    # Initialize model
    print("\nStep 3: Initializing model...")
    print(f"Creating model with num_classes={num_classes}")
    if args.char_model:
        try:
            print("Initializing CRAFT model...")
            # Try with reduced precision to save memory
            use_fp16 = True if device.type == 'cuda' else False
            
            # Clear cache before initialization
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            model = CRAFTFontClassifier(
                num_fonts=num_classes,
                device=device,
                patch_size=32,
                embedding_dim=args.embedding_dim,
                craft_fp16=use_fp16
            ).to(device)
            
            # print("\nTesting batch independence...")
            # char_model = model.font_classifier
            # check_char_model_batch_independence(char_model, device=device)
            
        except RuntimeError as e:
            
            if "CUDA" in str(e):
                print(f"CUDA error during model initialization: {e}")
                print("Trying with CPU for CRAFT model...")
                
                # Try with CPU for CRAFT but keep classifier on GPU if available
                craft_device = torch.device('cpu')
                model = CRAFTFontClassifier(
                    num_fonts=num_classes,
                    device=craft_device,
                    patch_size=32,
                    embedding_dim=args.embedding_dim,
                    craft_fp16=False
                )
                # Only move classifier to GPU
                if device.type == 'cuda':
                    model.font_classifier = model.font_classifier.to(device)
                model = model.to(device)
            else:
                raise e
       
       # TODO add some assertions to the model
    else:
        model = SimpleCNN(
            num_classes=num_classes,
            embedding_dim=args.embedding_dim,
            input_size=args.resolution,
            initial_channels=args.initial_channels
        ).to(device)

        actual_classes = model.classifier.weight.shape[0]
        assert actual_classes == num_classes, (
            f"Critical Error: Model initialized with wrong number of classes. "
            f"Got {actual_classes}, expected {num_classes}"
        )
        

        print(f"Model initialized with classifier shape: {model.classifier.weight.shape}")
        print(f"Number of classes from loader: {num_classes}")
        print(f"Model number of classes: {model.classifier.weight.shape[0]}")


    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params {total_params}')
    print(f'Trainable params {trainable_params}')
    count_parameters(model)

    
    # Initialize metrics calculator
    metrics_calculator = ClassificationMetrics(num_classes=num_classes, device=device)
    
    # Set up training
    wandb.watch(model, log_freq=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up schedulers
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs * len(train_loader)
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - warmup_epochs) * len(train_loader),
        eta_min=1e-6
    )
    
    print("Starting training...")
    best_test_acc = 0.0
    metrics_calculator.reset_timing()
    
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            warmup_epochs, warmup_scheduler, main_scheduler, metrics_calculator, args.char_model
        )
        
        # Evaluate
        test_metrics = evaluate(
            model, test_loader, criterion, device, metrics_calculator, epoch, args.char_model
        )
        
        # Combine metrics and add epoch info
        combined_metrics = {
            'epoch': epoch + 1,
            'epoch_time': time.time() - epoch_start_time,
            **train_metrics,
            **test_metrics
        }
        
        # Log all metrics
        wandb.log(combined_metrics)
        
        # Extract key metrics for printing and model saving
        train_loss = train_metrics['train/loss']
        test_loss = test_metrics['test/loss']
        train_acc = train_metrics['train/top1_acc']  # Updated to match new key
        test_acc = test_metrics['test/top1_acc']     # Updated to match new key
        
        # Print epoch summary
        print('\nEpoch Summary:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss:  {test_loss:.3f} | Test Acc:  {test_acc:.2f}%')
        print(f'Test Top-5 Acc: {test_metrics["test/top5_acc"]:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Only print top-5 acc if available
        if 'test/top5_acc' in test_metrics:
            print(f'Test Top-5 Acc: {test_metrics["test/top5_acc"]:.2f}%')
        
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best model! (Test Acc: {test_acc:.2f}%)")
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'num_classes': num_classes
            }

            # Save best model
            # model_name = f"fontCNN_BS{args.batch_size}\
            #     -ED{args.embedding_dim}-IC{args.initial_channels}.pt"
            # model_path = os.path.join(args.data_dir, model_name)
            model_path = get_model_path(
                base_dir=args.data_dir,
                prefix='fontCNN',
                batch_size=args.batch_size,
                embedding_dim=args.embedding_dim,
                initial_channels=args.initial_channels
            )
            # classifier_shape = best_model_state['model_state_dict']['classifier.weight'].shape
            # assert classifier_shape[0] == num_classes, (
            #     f"Critical Error: Attempting to save model with wrong number of classes. "
            #     f"Got {classifier_shape[0]}, expected {num_classes}"
            # )
            if args.char_model:
                # For character-based model, classifier is at font_classifier.font_classifier
                classifier_key = 'font_classifier.font_classifier.weight'
            else:
                # For simple CNN, classifier is directly at classifier
                classifier_key = 'classifier.weight'

            # Try to get the classifier shape
            if classifier_key in best_model_state['model_state_dict']:
                classifier_shape = best_model_state['model_state_dict'][classifier_key].shape
                print(f"Found classifier at {classifier_key} with shape {classifier_shape}")
                
                # Verify number of classes
                assert classifier_shape[0] == num_classes, (
                    f"Critical Error: Attempting to save model with wrong number of classes. "
                    f"Got {classifier_shape[0]}, expected {num_classes}"
                )
            else:
                # If key not found, print available keys for debugging
                print("Classifier key not found. Available keys in state_dict:")
                for key in best_model_state['model_state_dict'].keys():
                    if 'weight' in key and key.endswith('weight'):
                        print(f"  {key}: {best_model_state['model_state_dict'][key].shape}")
                
                print(f"WARNING: Could not verify classifier shape for {num_classes} classes")
            
            torch.save(best_model_state, model_path)
            print(f"Saved checkpoint with classifier shape: {classifier_shape}")

            
        # Update wandb summary periodically
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            wandb.run.summary.update({
                "best_test_acc": best_test_acc,
                "final_train_loss": train_loss,
                "final_test_loss": test_loss,
                "epochs_to_best": epoch + 1,
                "train_test_gap": abs(train_loss - test_loss),
                "final_train_test_acc_gap": abs(train_acc - test_acc)
            })
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print("Calculating class embeddings for each font using the training data")

    # class_embeddings = compute_class_embeddings(
    #     model, train_loader, num_classes, device
    # )
    
    # # Add embeddings to the best model state and save
    # best_model_state['class_embeddings'] = class_embeddings
    # torch.save(best_model_state, 'best_model.pt')


if __name__ == "__main__":
    main()

