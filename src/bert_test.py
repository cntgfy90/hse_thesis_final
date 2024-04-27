import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    hamming_loss,
)


def bert_test(model, validation_loader, loss_fn, num_classes, device="cpu"):
    total_obs = 0
    correct_predictions = np.zeros((num_classes))

    losses = []
    hl = []
    macro_precision = []
    micro_precision = []
    macro_recall = []
    micro_recall = []
    accuracy_subset = []

    # set model to eval mode (turn off dropout, fix batch norm)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader, 0):
            if ((batch_idx + 1) % 100) == 0:
                print(f"Batch: {batch_idx + 1}")
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()

            total_obs += outputs.shape[0]
            correct_predictions += np.sum(outputs.T == targets.T, axis=1)

            # Hamming loss
            hl.append(hamming_loss(targets, outputs))

            accuracy_subset.append(accuracy_score(targets, outputs))

            # Macro / mictor precision
            macro_precision.append(
                precision_score(targets, outputs, average="macro", zero_division=1)
            )
            micro_precision.append(
                precision_score(targets, outputs, average="micro", zero_division=1)
            )

            # Macro / mictor recall
            macro_recall.append(
                recall_score(targets, outputs, average="macro", zero_division=1)
            )
            micro_recall.append(
                recall_score(targets, outputs, average="micro", zero_division=1)
            )

    adapated_accuracy_score = np.sum(correct_predictions / total_obs) / num_classes

    return {
        "accuracy_adapated": adapated_accuracy_score,
        "accuracy_subset": accuracy_subset,
        "correct_predictions": correct_predictions,
        "loss": losses,
        "hamming_loss": hl,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
    }
