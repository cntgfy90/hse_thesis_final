import torch
from transformers import DebertaModel


class DeBERTa(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.fc = torch.nn.Linear(768, n_classes)

    def forward(self, ids, mask, token_type_ids):
        output = self.deberta(ids, attention_mask=mask, token_type_ids=token_type_ids)
        return self.fc(output[0][:, 0, :].squeeze(1))
