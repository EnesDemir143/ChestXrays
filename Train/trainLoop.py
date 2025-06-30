import torch
from .logger import app_logger
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score
)

def train_model(model, data_loader, validation_loader, device, loss_fn, optimiser, model_save_path, early_stopping_patience=5, EPOCHS=30):
    model.to(device)
    app_logger.info(f'Starting training for {EPOCHS} epochs')

    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    accuracies = [] 

    os.makedirs("roc_curves", exist_ok=True)  

    for epoch in range(EPOCHS):
        model.train()

        all_preds = []
        all_labels = []
        all_probs = [] 

        total_loss = 0 
        batch_count = 0

        app_logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        app_logger.info("-" * 15)

        train_loop = tqdm(data_loader, desc="Training", leave=False)
        for input, label in train_loop:
            inputs, labels = input.to(device), label.to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            batch_count += 1
            if batch_count % 10 == 0:
                app_logger.info(f"Batch {batch_count}: Loss = {loss.item():.4f}")

        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            val_loop = tqdm(validation_loader, desc="Validating", leave=False)
            for val_input, val_label in val_loop:
                val_inputs, val_labels = val_input.to(device), val_label.to(device)
                val_predictions = model(val_inputs)
                val_loss = loss_fn(val_predictions, val_labels)
                total_val_loss += val_loss.item()
                val_batch_count += 1

                probs = torch.softmax(val_predictions, dim=1)[:, 1]
                preds = torch.argmax(val_predictions, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
        val_losses.append(avg_val_loss)

        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
            app_logger.warning("ROC AUC couldn't be calculated. Probably only one class present.")

        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - Epoch {epoch+1}")
            plt.legend()
            plt.savefig(f"roc_curves/epoch_{epoch+1}_roc.png")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='o')
            plt.title('Train vs Validation Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig("loss_over_epochs.png")
            plt.close()
            
        except ValueError:
            app_logger.warning("ROC curve couldn't be plotted due to invalid data.")
        


        app_logger.info(f"Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")
        

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()  

        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        app_logger.info(
        f"Confusion Matrix:\n{cm.tolist()}\n"
        f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} | "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}"
    )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            app_logger.info(f" Validation loss improved. Model saved to {model_save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            app_logger.info(f" No improvement. Early stopping counter: {early_stop_counter}/{early_stopping_patience}")

        if early_stop_counter >= early_stopping_patience:
            app_logger.info(" Early stopping triggered. Stopping training.")
            break

    print("\nTraining completed!")
