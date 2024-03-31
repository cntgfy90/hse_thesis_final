import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, hamming_loss

def bert_test(model, validation_loader, loss_fn, device='cpu'):
    losses = []
    hl = []
    macro_precision = []
    micro_precision = []
    macro_recall = []
    micro_recall = []
    correct_predictions = 0
    num_samples = 0
    # set model to eval mode (turn off dropout, fix batch norm)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader, 0):
            if ((batch_idx + 1) % 100) == 0:
                print(f"Batch: {batch_idx + 1}")
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # validation accuracy
            # add sigmoid
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs==targets)
            # total number of elements in the 2D array
            num_samples += targets.size

            # Hamming loss
            hl.append(hamming_loss(targets, outputs))

            # Macro / mictor precision
            macro_precision.append(precision_score(targets, outputs, average='macro'))
            micro_precision.append(precision_score(targets, outputs, average='micro'))

            # Macro / mictor recall
            macro_recall.append(recall_score(targets, outputs, average='macro'))
            micro_recall.append(recall_score(targets, outputs, average='micro'))

    return {
        'accuracy': float(correct_predictions)/num_samples,
        'bce_loss': losses,
        'hamming_loss': hl,
        'macro_precision': macro_precision,
        'micro_precision': micro_precision,
        'macro_recall': macro_recall,
        'micro_recall': micro_recall,
    }