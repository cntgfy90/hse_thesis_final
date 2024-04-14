import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    hamming_loss,
)
from utils import get_accuracy


def bert_test(model, validation_loader, loss_fn, device="cpu"):
    losses = []
    hl = []
    macro_precision = []
    micro_precision = []
    macro_recall = []
    micro_recall = []
    accuracy = []
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

            # Hamming loss
            hl.append(hamming_loss(targets, outputs))

            accuracy.append(get_accuracy(outputs, targets))
            accuracy_subset.append(accuracy_score(targets, outputs))

            # Macro / mictor precision
            macro_precision.append(precision_score(targets, outputs, average="macro", zero_division=1))
            micro_precision.append(precision_score(targets, outputs, average="micro", zero_division=1))

            # Macro / mictor recall
            macro_recall.append(recall_score(targets, outputs, average="macro", zero_division=1))
            micro_recall.append(recall_score(targets, outputs, average="micro", zero_division=1))

    return {
        "accuracy": accuracy,
        "accuracy_subset": accuracy_subset,
        "loss": losses,
        "hamming_loss": hl,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
    }
