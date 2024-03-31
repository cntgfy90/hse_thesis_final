import torch

def bert_train(model, train_loader, optimizer, loss_fn, epoch, device='cpu'):
    model.train()
    for _, data in enumerate(train_loader, 0):
        print(f"Batch: {_}")
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        targets = data['targets'].to(device)

        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)

        if _%50 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # grad descent step
        optimizer.step()