# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 09:32:12 2024

This script contains the LWM pre-training and task-specific fine-tuning functions.

@author: Sadjad Alikhani
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv
from utils import count_parameters
import time
#%% LOSS FUNCTION
def nmse_loss(y_pred, y_true):
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)
    mse = torch.sum((y_true_flat - y_pred_flat)**2, dim=-1)
    normalization = torch.sum(y_true_flat**2, dim=-1)
    return mse / normalization
#%%
def train_lwm(model, train_loaders, val_loaders, optimizer, scheduler, epochs, device, save_dir="models", log_file="training_log.csv"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize CSV log
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train NMSE", "Validation NMSE", "Learning Rate", "Best Model"])

    train_nmse_losses = []
    val_nmse_losses = []
    best_val_nmse = float('inf')

    for epoch in range(epochs):
        model.train()
        train_nmse = 0.0
        train_samples = 0

        # Training loop across all buckets
        print(f"\nEpoch {epoch + 1}/{epochs} [Training]")
        for length, train_loader in train_loaders.items():
            print(f"Processing sequences of length {length}")
            with tqdm(train_loader, desc=f"Length {length} [Training]", unit="batch") as t:
                for batch in t:
                    # train_batches += 1
                    optimizer.zero_grad()

                    # Move data to device
                    input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                    # Forward pass
                    logits_lm, _, _ = model(input_ids, masked_pos)

                    # Compute NMSE
                    loss = torch.sum(nmse_loss(masked_tokens, logits_lm))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    train_nmse += loss.item()
                    train_samples += input_ids.shape[0]

                    # Update progress bar
                    t.set_postfix({"nmse": train_nmse/train_samples, "lr": scheduler.get_last_lr()[0]})

        # Average NMSE across training batches
        train_nmse /= max(train_samples, 1)
        train_nmse_losses.append(train_nmse)
        
        if epoch % 2 == 0:
            # Validation loop across all buckets
            model.eval()
            val_nmse = 0.0
            val_samples = 0
            with torch.no_grad():
                print(f"\nEpoch {epoch + 1}/{epochs} [Validation]")
                for length, val_loader in val_loaders.items():
                    print(f"Processing sequences of length {length}")
                    with tqdm(val_loader, desc=f"Length {length} [Validation]", unit="batch") as t:
                        for batch in t:
                            # val_batches += 1
        
                            # Move data to device
                            input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]
        
                            # Forward pass
                            logits_lm, _, _ = model(input_ids, masked_pos)
        
                            # Compute NMSE
                            loss = torch.sum(nmse_loss(masked_tokens, logits_lm))
                            val_nmse += loss.item()
                            val_samples += input_ids.shape[0]
        
                            # Update progress bar
                            t.set_postfix({"nmse": val_nmse/val_samples})
    
            # Average NMSE across validation batches
            val_nmse /= max(val_samples, 1)
            val_nmse_losses.append(val_nmse)

            # Save model if validation NMSE improves
            is_best_model = False
            if val_nmse < best_val_nmse:
                best_val_nmse = val_nmse
                model_path = os.path.join(save_dir, f"lwm_epoch{epoch+1}_train{train_nmse:.4f}_val{val_nmse:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved: {model_path}")
                is_best_model = True

        # Log the results
        print(f"  Train NMSE: {train_nmse:.4f}")
        print(f"  Validation NMSE: {val_nmse:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # Append to CSV log
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_nmse, val_nmse, scheduler.get_last_lr()[0], is_best_model])

        # Plot losses after each epoch
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_nmse_losses) + 1), train_nmse_losses, label="Train NMSE")
        plt.plot(range(1, len(val_nmse_losses) + 1), val_nmse_losses, label="Validation NMSE")
        plt.xlabel("Epochs")
        plt.ylabel("NMSE")
        plt.title("Training and Validation NMSE Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Training and validation complete.")
    return model
#%% FINE-TUNE
from torch.cuda.amp import GradScaler, autocast

# Define the ClassificationHead
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Define the RegressionHead
class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

class CustomClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        
        super().__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(input_dim, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),                 
            # nn.Dropout(0.1),           
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class CustomRegressionHead(nn.Module):
    def __init__(self, input_dim=4096, output_dim=1024, num_heads=1, dropout=0.1):
        super().__init__()
        self.linear1 = nn.utils.weight_norm(nn.Linear(input_dim, 2048))
        self.bn1 = nn.LayerNorm(input_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(2048, output_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(output_dim))
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        #residual = x  # 원본 유지
        residual = self.proj(x)  # [B, S, 1024]
        x = self.bn1(x)
        x = self.linear1(x)        # [B, 2048]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) + self.decoder_bias  # [B, output_dim]
        x = x + residual  # 최종 residual 연결
        return x

def custom_heads(input_dim, num_classes=None, output_dim=None, task_type="classification"):
    """
    Creates a custom head for classification or regression tasks.
    Users should modify the class implementations for further customization.

    Args:
        input_dim (int): Input dimension of the head.
        num_classes (int): Number of classes for classification tasks. Ignored for regression.
        task_type (str): "classification" or "regression".

    Returns:
        nn.Module: Custom head for the specified task.
    """
    if task_type == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks.")
        return CustomClassificationHead(input_dim=input_dim, num_classes=num_classes)
    elif task_type == "regression":
        return CustomRegressionHead(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")
#%%
# Fine-tuning wrapper for the base model
class FineTuningWrapper(nn.Module):
    def __init__(self, model, task_head, mask, fine_tune_layers="full"):
        super().__init__()
        self.model = model
        self.task_head = task_head
        ##(HJ)
        self.mask = mask
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Handle fine-tuning layers
        if fine_tune_layers is not None:
            if fine_tune_layers == "full":
                # Unfreeze all layers if "all" is specified
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                # Get a list of all available layer names in the model
                available_layers = [name for name, _ in self.model.named_parameters()]
                
                # Validate that specified layers exist in the model
                for layer in fine_tune_layers:
                    if not any(layer in lname for lname in available_layers):
                        raise ValueError(
                            f"Layer '{layer}' not found in the model. "
                            f"Available layers: {available_layers}"
                        )
                
                # Unfreeze only the specified layers
                for name, param in self.model.named_parameters():
                    if any(layer in name for layer in fine_tune_layers):
                        param.requires_grad = True

    def forward(self, x, input_type="cls_emb", mask = False):
        if input_type == "raw":
            task_input = x.view(x.size(0), -1)
        else:
            embeddings, attn_maps = self.model(x, input_type=input_type, mask = mask)  # Get embeddings from the base model
            if input_type == "cls_emb":
                task_input = embeddings[:, 0, :]  # CLS token
            elif input_type == "channel_emb":
                chs_emb = embeddings[:, 1:, :]
                task_input = chs_emb.view(chs_emb.size(0), -1) # embeddings.mean(dim=1)  # Mean pooling over channel embeddings
        return self.task_head(task_input), 0 if input_type=="raw" else attn_maps
#%%
# Fine-tuning function
from sklearn.metrics import f1_score
def finetune(
    base_model,
    train_loader,
    val_loader=None, 
    task_type="classification",
    input_type="cls_emb",
    num_classes=None,
    output_dim=None,
    use_custom_head=False,
    fine_tune_layers=None,
    optimizer_config=None,
    criterion=None,
    mask=False,
    epochs=10,
    fine_tune=False,
    resume_path = None,
    device="cuda",
    task="Beam Prediction"
):
    """
    Configures and fine-tunes the base model with user-defined settings, saving results and models.
    """
    # Create results folder
    time_now = f"{time.time():.0f}"
    results_folder = f"results/{task}/{time_now}"
    os.makedirs(results_folder, exist_ok=True)
    log_file = os.path.join(results_folder, "training_log.csv")

    # Initialize the CSV log
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Task", "Input", "Epoch", "Train Loss", "Validation Loss", "F1-Score (Classification)", "Learning Rate", "Time"])

#    (HJ)
    if train_loader is not None:
        for batch in train_loader:
            input_data, targets = batch[0].to(device), batch[1].to(device)
            break
    else:
        attn_maps = None
        input_data = torch.randn(64, 33, 32).to(device)
    
    if input_type == "cls_emb":
        n_patches = 1
        patch_size = 128
    elif input_type == "channel_emb":
        n_patches = input_data.shape[1] - 1
        patch_size = 128
        # # (HJ)
        # if mask == True:
        #     patch_size = 32
        # else:
        #     patch_size = 128
    elif input_type == "raw":
        n_patches = input_data.shape[1]
        patch_size = 32
        # patch_size = 1
    if use_custom_head:
        custom_head = custom_heads(input_dim=n_patches*patch_size, 
                                   num_classes=num_classes, 
                                   output_dim=output_dim,
                                   task_type=task_type)
        
    # Handle DataParallel models
    if isinstance(base_model, nn.DataParallel):
        base_model = base_model.module

    # Set up the task-specific head
    if use_custom_head:
        task_head = custom_head
    elif task_type == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks.")
        task_head = ClassificationHead(input_dim=n_patches*patch_size, num_classes=num_classes) # input_dim=base_model.embedding.d_model
    elif task_type == "regression":
        task_head = RegressionHead(input_dim=n_patches*patch_size) # input_dim=base_model.embedding.d_model
    else:
        raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")

    # Wrap the model with the fine-tuning head
    wrapper = FineTuningWrapper(base_model, task_head, mask, fine_tune_layers=fine_tune_layers)
    wrapper = wrapper.to(device)
    
    print(f'Number of head parameters: {count_parameters(wrapper)}')
    
   # Set default optimizer config if not provided
    if optimizer_config is None:
        optimizer_config = {"lr": 1e-4}
    # Set up the optimizer
    optimizer = torch.optim.Adam(wrapper.parameters(), **optimizer_config)
    # Set up the scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)  # Example: Reduce LR by 10x every 10 epochs

    # Set up the loss criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()

    scaler = GradScaler()
    train_losses, val_losses, Accuracy = [], [], []
    best_val_loss = float("inf")
    best_model_path = None
    attn_maps = None

    if resume_path is not None and os.path.exists(resume_path):
        wrapper.load_state_dict(torch.load(resume_path, map_location=device))
        best_model_path = resume_path
        print(f"✅ Resuming from the resume_path: {resume_path}")
    else:
        print("🚀 Starting training from scratch")

    if (fine_tune):

        for epoch in range(epochs):
            # Training loop
            wrapper.train()
            epoch_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as progress_bar:
                for batch in progress_bar:
                    input_data, targets = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()

                    #diff = torch.abs(input_data[:, 1:, :].view(input_data.size(0), -1) - targets)
                    #print("(T) Mean absolute difference:", torch.mean(diff))

                    with autocast():
                        outputs, attn_maps = wrapper(input_data, input_type=input_type)
                        loss = criterion(outputs, targets)

                    #diff = torch.abs(outputs - targets)
                    #print("(T) Mean absolute difference:", torch.mean(diff))

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"Loss": loss.item()})

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation loop
            if val_loader:
                wrapper.eval()
                val_loss = 0.0
                all_preds, all_targets = [], []

                with torch.no_grad():
                    for batch in val_loader:
                        input_data, targets = batch[0].to(device), batch[1].to(device)
                        with autocast():
                            outputs, _ = wrapper(input_data, input_type=input_type)
                            loss = criterion(outputs, targets)

                        val_loss += loss.item()

                        if task_type == "classification":
                            preds = torch.argmax(outputs, dim=1).cpu().numpy()
                            all_preds.extend(preds)
                            all_targets.extend(targets.cpu().numpy())

                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                time_now = f"{time.time():.0f}"
                # Save the best model
                if (epoch>epochs-5) and (avg_val_loss < best_val_loss):
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(results_folder, f"{input_type}_epoch{epoch+1}_valLoss{avg_val_loss:.4f}_{time_now}.pth")
                    best_model_path = best_model_path.replace("\\", "/")
                    torch.save(wrapper.state_dict(), best_model_path)
                    print(f"Model saved at {best_model_path} with validation loss: {best_val_loss:.4f}")

                # Compute F1-score for classification tasks
                f1 = None
                if task_type == "classification":
                    f1 = f1_score(all_targets, all_preds, average="weighted")
                    print(f"Epoch {epoch + 1}, Validation F1-Score: {f1:.4f}")
                    Accuracy.append(f1)
                elif task_type == "regression":
                    preds = outputs.detach().cpu()
                    trues = targets.detach().cpu()
                    mse = torch.sum((preds - trues) ** 2, dim=-1)
                    power = torch.sum(trues ** 2, dim=-1) + 1e-10
                    nmse = torch.mean(mse / power).item()
                    print(f"Epoch {epoch + 1}, Validation NMSE: {nmse:.6f}")
                    #Accuracy.append(nmse)
            scheduler.step()

            # Log results
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([task, input_type, epoch + 1, avg_train_loss, avg_val_loss, f1 if f1 is not None else "-", scheduler.get_last_lr()[0], f"{time_now}"])

    else:
        print("!!! Inference!!!")
        wrapper.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Inference"):
                input_data, targets = batch[0].to(device), batch[1].to(device)
                with autocast():
                    outputs, _ = wrapper(input_data, input_type=input_type)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        time_now = f"{time.time():.0f}"
        print(f"[Evaluation only] Validation loss: {avg_val_loss:.4f}")

    if task_type == "regression":
        preds = outputs.detach().cpu()
        trues = targets.detach().cpu()
        mse = torch.sum((preds - trues) ** 2, dim=-1)
        power = torch.sum(trues ** 2, dim=-1) + 1e-10
        nmse = torch.mean(mse / power).item()
        print(f"[Evaluation only] Validation NMSE: {nmse:.6f}")
        Accuracy.append(nmse)

        # #(HJ)추가수정: Accruacy + Loss 통합
        # # Plot training and validation losses + accuracy/NMSE
        # fig, ax1 = plt.subplots(figsize=(10, 6))
        #
        # # 왼쪽 Y축: Loss
        # ax1.plot(range(1, epochs + 1), train_losses, label="Training Loss", color="tab:blue")
        # ax1.plot(range(1, epochs + 1), val_losses, label="Validation Loss", linestyle="--", color="tab:orange")
        # ax1.set_xlabel("Epochs")
        # ax1.set_ylabel("Loss", color="tab:blue")
        # ax1.tick_params(axis='y', labelcolor="tab:blue")
        # ax1.grid(True)
        #
        # # 오른쪽 Y축: Accuracy or NMSE
        # ax2 = ax1.twinx()
        # ax2.plot(
        #     range(1, epochs + 1), Accuracy,
        #     label="F1-Score" if task_type == "classification" else "NMSE",
        #     linestyle="-.", marker="o", color="tab:green"
        # )
        # ax2.set_ylabel("F1-Score" if task_type == "classification" else "NMSE", color="tab:green")
        # ax2.tick_params(axis='y', labelcolor="tab:green")
        #
        # # 범례 통합
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        #
        # # 제목 및 출력
        # plt.title("Training/Validation Loss and " + ("F1-Score" if task_type == "classification" else "NMSE"))
        # plt.tight_layout()
        #
        # # 저장도 가능
        # # plt.savefig(os.path.join(results_folder, "loss_accuracy_curve.png"))
        # plt.show()

    return wrapper, best_model_path, train_losses, val_losses, Accuracy, attn_maps