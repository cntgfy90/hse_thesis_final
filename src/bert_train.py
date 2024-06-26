import torch
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
)
import numpy as np


def bert_train(
    model, train_loader, optimizer, loss_fn, epochs, num_classes, device="cpu"
):
    model.train()

    losses = []
    correct_predictions = []
    accuracy_adapted = []
    accuracy_subset = []
    hl = []

    for epoch in range(epochs):
        total_obs = 0
        epoch_correct_predictions = np.zeros((num_classes))

        epoch_losses = []
        epoch_accuracy_subset = []
        epoch_hl = []

        for batch_idx, data in enumerate(train_loader, 0):
            if ((batch_idx + 1) % 500) == 0:
                print(f"Batch: {batch_idx + 1}")

            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            targets = data["targets"].to(device)

            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            epoch_losses.append(loss)

            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()

            total_obs += outputs.shape[0]
            epoch_correct_predictions += np.sum(outputs.T == targets.T, axis=1)

            epoch_accuracy_subset.append(accuracy_score(targets, outputs))
            epoch_hl.append(hamming_loss(targets, outputs))

            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch: {epoch}, Loss:  {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # grad descent step
            optimizer.step()

        losses.append(epoch_losses)
        correct_predictions.append(epoch_correct_predictions)
        adapated_accuracy_score = np.sum(epoch_correct_predictions / total_obs) / num_classes
        accuracy_adapted.append(adapated_accuracy_score)
        accuracy_subset.append(epoch_accuracy_subset)
        hl.append(epoch_hl)

    return {
        "loss": losses,
        "correct_predictions": correct_predictions,
        "accuracy_adapted": accuracy_adapted,
        "accuracy_subset": accuracy_subset,
        "hl": hl,
    }
