from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import torch


def train_model(model, train_loader, optimizer, scheduler, epochs=1,
                accumulation_steps=4, model_save_path='../models/bert_for_sequence_classification'):

    # Move the model to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize the gradient scaler for mixed precision training
    scaler = GradScaler()

    # Set the model to training mode
    model.train()

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)  # Initialize tqdm
        for step, batch in enumerate(loop):
            # Reset gradients on the optimizer
            optimizer.zero_grad()

            # Unpack the batch and move to the GPU if available
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]

            # Perform a forward pass in mixed precision
            with autocast():
                outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels.long())
                loss = outputs.loss / accumulation_steps  # Normalize the loss to account for accumulation

            # Backpropagate in mixed precision
            scaler.scale(loss).backward()

            # Perform gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                # Perform an optimization step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # Update the learning rate

                # Update the tqdm loop
                loop.set_description(f'Epoch {epoch+1}/{epochs}')
                loop.set_postfix(loss=loss.item())

    # Save the model
    model.save_pretrained(model_save_path)


def evaluate_model(model, val_loader):
    # Set the model to evaluation mode
    model.eval()
    val_predictions = []
    val_true_labels = []

    # Assume the data loader also returns attention masks and update the unpacking accordingly
    with torch.no_grad():
        for batch in val_loader:
            # Update here to unpack three values
            batch_input_ids, batch_attention_mask, batch_labels = batch
            # Move batch to GPU
            batch_input_ids = batch_input_ids.to(model.device)
            batch_attention_mask = batch_attention_mask.to(model.device)
            batch_labels = batch_labels.to(model.device)

            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            val_predictions.extend(predictions.cpu().numpy())
            val_true_labels.extend(batch_labels.cpu().numpy())

    # Calculate the accuracy
    accuracy = accuracy_score(val_true_labels, val_predictions)
    print(f'Validation Accuracy: {accuracy}')
    return accuracy
