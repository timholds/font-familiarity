import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from dataset import get_dataloaders, get_char_dataloaders
from font_model import SimpleCNN
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
                warmup_epochs, warmup_scheduler, main_scheduler,
                metrics_calculator, char_model=False, model_name=None, 
                contrastive_weight=None, auxiliary_weight=None, total_samples_seen=0):
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
            targets = batch_data['labels'].to(device)
            batch_size = targets.size(0)

            # Check if we're using precomputed patches
            if 'patches' in batch_data and 'attention_mask' in batch_data:
                # We have precomputed patches - move to device
                batch_data['patches'] = batch_data['patches'].to(device)
                batch_data['attention_mask'] = batch_data['attention_mask'].to(device)
            elif 'images' in batch_data:
                # Traditional path with images
                batch_data['images'] = batch_data['images'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_data)
            
            # Handle different output formats from model
            if isinstance(outputs, tuple):
                logits, _ = outputs
            elif isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # Handle loss calculation
            if contrastive_weight is not None:
                # Contrastive loss with character models
                categories = batch_data['category'].to(device)
                embeddings = model.get_embedding(batch_data)
                
                cce_loss = nn.CrossEntropyLoss()(logits, targets)
                contrastive_loss = criterion(embeddings, categories, targets)
                loss = cce_loss + contrastive_weight * contrastive_loss
                
                # DEBUG: Check if categories are being loaded properly
                if batch_idx == 0:
                    print(f"\nDEBUG - Batch {batch_idx}:")
                    print(f"  Categories shape: {categories.shape}, unique values: {categories.unique().tolist()}")
                    print(f"  Targets (font labels) shape: {targets.shape}, sample values: {targets[:5].tolist()}")
                    print(f"  Embeddings shape: {embeddings.shape}")
                    print(f"  Contrastive loss value: {contrastive_loss.item()}")
            elif auxiliary_weight is not None:
                # Auxiliary category loss with character models
                categories = batch_data['category'].to(device)
                embeddings = model.get_embedding(batch_data)
                
                cce_loss = nn.CrossEntropyLoss()(logits, targets)
                auxiliary_loss = criterion(embeddings, categories, targets)
                loss = cce_loss + auxiliary_weight * auxiliary_loss
            else:
                # Regular cross-entropy loss
                loss = criterion(logits, targets)

            # Backward pass and optimization
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            # Non-character model (simplified - not the main focus)
            data, targets = batch_data
            data, targets = data.to(device), targets.to(device)
            batch_size = targets.size(0)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            
        # just do a couple batches
        if batch_idx == 1 or batch_idx == 10:
            # Visualize a few samples from the batch
            debug_path = os.path.splitext(model_name)[0]
            vis_path  = f"debug/{debug_path}/char_preds_epoch_{epoch}_batch_{batch_idx}"
            vis_path2 = f"debug/{debug_path}/extract_craft_epoch_{epoch}_batch_{batch_idx}"
            vis_path3 = f"debug/{debug_path}/craft_detections_epoch_{epoch}_batch_{batch_idx}"

            # Extract patches for visualization - handle both cases
            if 'patches' in batch_data and char_model:
                patches = batch_data['patches'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                
                # If model has visualization method, use it
                # TODO why is padding not showing up in these visualizations?

                if hasattr(model, 'visualize_char_preds'):
                    with torch.no_grad():
                        model.visualize_char_preds(
                            patches=patches,
                            attention_mask=attention_mask,
                            predictions=pred,
                            targets=targets,
                            save_path=vis_path
                        )
            elif char_model and hasattr(model, 'extract_patches_with_craft') and not model.use_precomputed_craft:
                # Only try to extract patches if not using precomputed and CRAFT is available
                with torch.no_grad(): 
                    patch_data = model.extract_patches_with_craft(images) # images BHWC
                    model.visualize_char_preds(
                        patches=patch_data['patches'],
                        attention_mask=patch_data['attention_mask'],
                        predictions=pred,
                        targets=targets,
                        save_path=vis_path2
                    )

            # Add CRAFT detection visualization - only if not using precomputed
            # bc if we are using precompute, data loader is not returning original images, just patches
            if char_model and hasattr(model, 'craft') and hasattr(model, 'visualize_craft_detections') and not model.use_precomputed_craft:
                model.visualize_craft_detections(
                    images=images,  # torch tensor BHWC 0, 255
                    label_mapping=train_loader.dataset.label_mapping,
                    targets=targets,
                    save_path=vis_path3
                )

            # Compute and log batch metrics
            if batch_idx % 50 == 0:
                batch_metrics = {
                    'train/batch_loss': loss.item(),
                    'train/running_loss': running_loss,
                    'train/running_acc': current_acc,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/grad_norm': grad_norm.item(),
                    'total_samples': total_samples_seen + total_samples
                }
                wandb.log(batch_metrics)
            
            # Log both losses separately to monitor
            if contrastive_weight is not None:
                contrastive_to_class_ratio = contrastive_loss.item() / (cce_loss.item() + 1e-8)
                wandb.log({
                    'train/font_class_loss': cce_loss.item(),
                    'train/category_contrastive_loss': contrastive_loss.item(),
                    'train/contrastive_to_class_ratio': contrastive_to_class_ratio,
                    'train/total_loss': loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/grad_norm': grad_norm.item(),
                    'total_samples': total_samples_seen + total_samples
                })
            elif auxiliary_weight is not None:
                wandb.log({
                    'train/font_class_loss': cce_loss.item(),
                    'train/category_auxiliary_loss': auxiliary_loss.item(),
                    'train/total_loss': loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/grad_norm': grad_norm.item(),
                    'total_samples': total_samples_seen + total_samples
                })

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
    }
    
    return train_metrics, total_samples

def evaluate(model, test_loader, criterion, device, metrics_calculator, epoch=None, char_model=False, contrastive_weight=None):
    """Evaluate the model, supporting both character-based and whole-image models."""
    model.eval()
    test_loss = 0
    all_logits = []
    all_targets = []
    
    # Basic accuracy tracking
    correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc='Evaluating'):
            # Handle different data formats based on model type
            if char_model:
                # For character-based model with CRAFT
                
                # Move labels to device (should always be present)
                targets = batch_data['labels'].to(device)
                
                # Check if we're using precomputed patches or images
                if 'patches' in batch_data and 'attention_mask' in batch_data:
                    # Using precomputed patches - move to device
                    batch_data['patches'] = batch_data['patches'].to(device)
                    batch_data['attention_mask'] = batch_data['attention_mask'].to(device)
                elif 'images' in batch_data:
                    # Using images - move to device
                    batch_data['images'] = batch_data['images'].to(device)
                else:
                    raise ValueError("Batch data must contain either 'patches' or 'images'")

                # Forward pass with the batch data
                outputs = model(batch_data)
                
                # Extract logits from outputs
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
            else:
                # Original code for non-character models
                data, targets = batch_data
                data, targets = data.to(device), targets.to(device)
                logits = model(data)

            # Accumulate tensors for advanced metrics
            all_logits.append(logits)
            all_targets.append(targets)
            
            # Compute loss for logging - evaluation should only use cross-entropy
            test_loss += nn.CrossEntropyLoss()(logits, targets).item()
            
            # Top-1 accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            
            # Top-3 accuracy
            _, pred3 = logits.topk(3, 1, True, True)
            target_expanded3 = targets.view(-1, 1).expand_as(pred3)
            correct3 = pred3.eq(target_expanded3).any(dim=1).sum().item()
            top3_correct += correct3
            
            # Top-5 accuracy
            _, pred5 = logits.topk(5, 1, True, True)
            target_expanded = targets.view(-1, 1).expand_as(pred5)
            correct5 = pred5.eq(target_expanded).any(dim=1).sum().item()
            top5_correct += correct5
            
            total += targets.size(0)
    
    # Compute basic metrics
    avg_loss = test_loss / len(test_loader)
    top1_acc = 100. * correct / total
    top3_acc = 100. * top3_correct / total
    top5_acc = 100. * top5_correct / total
    
    # Initialize metrics dictionary with basic metrics
    test_metrics = {
        'test/loss': avg_loss,
        'test/top1_acc': top1_acc,
        'test/top3_acc': top3_acc,
        'test/top5_acc': top5_acc,
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
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--use_precomputed_craft", action="store_true", help="Use precomputed CRAFT results")
    parser.add_argument("--patch_size", type=int, default=32, help="Size of character patches")
    parser.add_argument("--n_attn_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--pad_x", type=int, default=0, help="Padding in x direction")
    parser.add_argument("--pad_y", type=int, default=0, help="Padding in y direction")
    parser.add_argument("--contrastive_weight", type=float, default=None, help="Weight for contrastive loss when blending with CCE. If provided, enables contrastive loss.")
    parser.add_argument("--auxiliary_weight", type=float, default=None, help="Weight for auxiliary category loss when blending with CCE. If provided, enables auxiliary loss.")
    parser.add_argument("--max_datapts", type=int, default=None, help="Maximum number of training/test datapoints to use (for quick testing)")
    parser.add_argument("--max_chars", type=int, default=100, help="Maximum number of character patches per image")

    args = parser.parse_args()
    warmup_epochs = max(args.epochs // 5, 1)

    torch.manual_seed(1)
    
    # Set up metrics logging
    wandb.init(
        project="Font-Familiarity",
        name=f"{time.strftime('%Y-%m-%d_%H-%M')}-BS{args.batch_size}-ED{args.embedding_dim}-IC{args.initial_channels}-PS{args.patch_size}",
        config={
            **vars(args),
            "architecture": "CNN",
            "warmup_epochs": warmup_epochs,
            "optimizer": "AdamW"
        }
    )
    
    # Define all metrics to use total_samples as x-axis
    wandb.define_metric("total_samples")
    wandb.define_metric("train/*", step_metric="total_samples")
    wandb.define_metric("test/*", step_metric="total_samples")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.pretrained_model:
        print(f"Loading checkpoint from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)

        # Extract checkpoint information
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['test_metrics']['test/top1_acc']
        num_classes = checkpoint['num_classes']
        
        print(f"Resuming from epoch {start_epoch}/{args.epochs} with previous best test accuracy {best_test_acc:.2f}%")
    else:
        start_epoch = 0

    # Load data
    print("Loading data...")
    if args.char_model:
        train_loader, test_loader, num_classes = get_char_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=os.cpu_count(),
            use_precomputed_craft=args.use_precomputed_craft,
            pad_x=args.pad_x,
            pad_y=args.pad_y,
            return_category=args.contrastive_weight is not None or args.auxiliary_weight is not None,
            max_datapts=args.max_datapts,
            max_chars=args.max_chars
        )
    else:
        if args.contrastive_weight is not None or args.auxiliary_weight is not None:
            # Use return_category=True in dataloader
            train_loader, test_loader, num_classes = get_dataloaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=os.cpu_count(),
                return_category=True
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
                patch_size=args.patch_size,
                embedding_dim=args.embedding_dim,
                initial_channels=args.initial_channels,
                n_attn_heads=args.n_attn_heads,
                craft_fp16=use_fp16,
                use_precomputed_craft=args.use_precomputed_craft,
                pad_x=args.pad_x,
                pad_y=args.pad_y,
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
                    initial_channels=args.initial_channels,
                    n_heads=args.n_attn_heads,
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

    if args.pretrained_model:
        # Load the full model from checkpoint
        if hasattr(checkpoint['model'], 'state_dict'):
            # If checkpoint contains full model, extract its architecture
            loaded_model = checkpoint['model']
            model = loaded_model
            print("Full model loaded from checkpoint")
        else:
            # Fallback to state dict loading (for backwards compatibility)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded from checkpoint")
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params {total_params}')
    print(f'Trainable params {trainable_params}')
    count_parameters(model)
    
    # Initialize metrics calculator
    metrics_calculator = ClassificationMetrics(num_classes=num_classes, device=device)
    
    # Set up training
    wandb.watch(model, log_freq=100)
    if args.contrastive_weight is not None:
        from contrastive_loss import OptimizedCategoryContrastiveLoss
        criterion = OptimizedCategoryContrastiveLoss()
    elif args.auxiliary_weight is not None:
        from contrastive_loss import SimpleCategoryLoss
        criterion = SimpleCategoryLoss(embedding_dim=args.embedding_dim, num_categories=5)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if args.pretrained_model:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded from checkpoint")

        # Calculate iterations completed so far
        total_iters_completed = start_epoch * len(train_loader)
        remaining_warmup_epochs = max(0, warmup_epochs - start_epoch)

        # Configure warmup scheduler
        if start_epoch < warmup_epochs:
            # Still in warmup phase - calculate current factor and remaining iters
            progress = start_epoch / warmup_epochs
            current_factor = 0.1 + 0.9 * progress  # Assuming start_factor=0.1
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=current_factor,
                end_factor=1.0,
                total_iters=remaining_warmup_epochs * len(train_loader)
            )
            print(f"Resuming warmup schedule at factor {current_factor:.4f} for {remaining_warmup_epochs} more epochs")
        else:
            # Warmup completed, create dummy scheduler
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=1.0,
                end_factor=1.0,
                total_iters=1
            )
            print("Warmup phase already completed")

        # Configure cosine scheduler
        if start_epoch >= warmup_epochs:
            # Already in cosine phase
            cosine_iters_completed = (start_epoch - warmup_epochs) * len(train_loader)
            total_cosine_iters = (args.epochs - warmup_epochs) * len(train_loader)
            # Create with proper last_epoch (-1 is the default for starting fresh)
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_cosine_iters,
                eta_min=1e-6,
                last_epoch=cosine_iters_completed - 1  # -1 because step() will be called again
            )
            print(f"Resuming cosine schedule after {cosine_iters_completed} iterations")
        else:
            # Not yet in cosine phase
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(args.epochs - warmup_epochs) * len(train_loader),
                eta_min=1e-6
            )
            print("Cosine phase not yet started")
            
        # Verify learning rate is correct by comparing with saved value
        if 'train_metrics' in checkpoint and 'train/learning_rate' in checkpoint['train_metrics']:
            checkpoint_lr = checkpoint['train_metrics']['train/learning_rate']
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Checkpoint LR: {checkpoint_lr:.6f}, Current LR: {current_lr:.6f}")
            
            # If there's a significant mismatch, force the LR
            if abs(checkpoint_lr - current_lr) > 1e-5:
                print(f"Warning: LR mismatch detected. Forcing LR to {checkpoint_lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = checkpoint_lr
    else:
        # Initialize schedulers fresh 
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
    total_samples_seen = 0  # Track total samples across all epochs

    model_path = get_model_path(
                base_dir=args.data_dir,
                prefix='fontCNN',
                batch_size=args.batch_size,
                embedding_dim=args.embedding_dim,
                initial_channels=args.initial_channels,
                patch_size=args.patch_size, 
                n_attn_heads=args.n_attn_heads,
            )
    
    # Create model-specific directory for checkpoints
    model_name_without_ext = os.path.splitext(os.path.basename(model_path))[0]
    checkpoint_dir = os.path.join(args.data_dir, model_name_without_ext)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints and best model will be saved to: {checkpoint_dir}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        epoch_start_time = time.time()
        # Train
        train_metrics, epoch_samples = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            warmup_epochs, warmup_scheduler, main_scheduler, 
            metrics_calculator, args.char_model, os.path.basename(model_path),
            contrastive_weight=args.contrastive_weight,
            auxiliary_weight=args.auxiliary_weight,
            total_samples_seen=total_samples_seen
        )
        total_samples_seen += epoch_samples
        
        # Evaluate
        test_metrics = evaluate(
            model, test_loader, criterion, device, metrics_calculator, epoch, args.char_model,
            contrastive_weight=args.contrastive_weight
        )
        
        # Combine metrics and add epoch info
        combined_metrics = {
            'epoch': epoch + 1,
            'epoch_time': time.time() - epoch_start_time,
            'total_samples': total_samples_seen,
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

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'num_classes': num_classes
        }, checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch+1} at {checkpoint_path}")
        
        # Save best model based on test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best model! (Test Acc: {test_acc:.2f}%)")
            best_model_state = {
                'epoch': epoch + 1,
                'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'num_classes': num_classes
            }

            # Save best model
            # model_name = f"fontCNN_BS{args.batch_size}\
            #     -ED{args.embedding_dim}-IC{args.initial_channels}.pt"
            # model_path = os.path.join(args.data_dir, model_name)
            
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

            # Try to get the classifier shape from the model's state dict
            model_state_dict = best_model_state['model'].state_dict()
            if classifier_key in model_state_dict:
                classifier_shape = model_state_dict[classifier_key].shape
                print(f"Found classifier at {classifier_key} with shape {classifier_shape}")
                
                # Verify number of classes
                assert classifier_shape[0] == num_classes, (
                    f"Critical Error: Attempting to save model with wrong number of classes. "
                    f"Got {classifier_shape[0]}, expected {num_classes}"
                )
            else:
                # If key not found, print available keys for debugging
                print("Classifier key not found. Available keys in state_dict:")
                for key in model_state_dict.keys():
                    if 'weight' in key and key.endswith('weight'):
                        print(f"  {key}: {model_state_dict[key].shape}")
                
                print(f"WARNING: Could not verify classifier shape for {num_classes} classes")
            

            # Save best model in the same model-specific directory
            best_model_path = os.path.join(checkpoint_dir, os.path.basename(model_path))
            torch.save(best_model_state, best_model_path)
            print(f"Saved best model at {best_model_path}")

            
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

