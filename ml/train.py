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
from utils import get_model_path

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
            # Batch data is a dictionary with patches, attention mask, and labels
            patches = batch_data['patches'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            targets = batch_data['labels'].to(device)
            batch_size = targets.size(0)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Different models might return different structures
            outputs = model(patches, attention_mask)
            
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
            if char_model:
                model.visualize_char_preds(data, batch_data['patches'], 
                                        batch_data['attention_mask'], 
                                        save_path=vis_path)

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
                # For character-based model
                patches = batch_data['patches'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                target = batch_data['labels'].to(device)
                
                # Forward pass with different possible output formats
                output = model(patches, attention_mask)
                if isinstance(output, tuple):
                    logits, _ = output  # Unpack (logits, attention_weights)
                elif isinstance(output, dict):
                    logits = output['logits']  # Extract from dictionary
                else:
                    logits = output  # Direct logits output
            else:
                # Traditional whole-image approach
                data, target = batch_data
                data, target = data.to(device), target.to(device)
                logits = model(data)
            
            # Accumulate tensors for advanced metrics
            all_logits.append(logits)
            all_targets.append(target)
            
            # Compute basic metrics on the fly
            test_loss += criterion(logits, target).item()
            
            # Top-1 accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            
            # Top-5 accuracy
            _, pred5 = logits.topk(5, 1, True, True)
            target_expanded = target.view(-1, 1).expand_as(pred5)
            correct5 = pred5.eq(target_expanded).any(dim=1).sum().item()
            top5_correct += correct5
            
            total += target.size(0)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/font_dataset_npz_test/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--initial_channels", type=int, default=16)
    parser.add_argument("--char_model", type=bool, default=True)
    
    args = parser.parse_args()
    warmup_epochs = max(args.epochs // 5, 1)
    
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
       model = CRAFTFontClassifier(
            num_fonts=num_classes,
            craft_weights_dir=args.craft_weights_dir,
            device=device,
            char_size=32,
            embedding_dim=args.embedding_dim,
            craft_fp16=args.craft_fp16
        ).to(device)
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
            classifier_shape = best_model_state['model_state_dict']['classifier.weight'].shape
            assert classifier_shape[0] == num_classes, (
                f"Critical Error: Attempting to save model with wrong number of classes. "
                f"Got {classifier_shape[0]}, expected {num_classes}"
            )
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

