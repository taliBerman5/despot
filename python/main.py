import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

NSTATES = 10
NACTIONS = 5


class NN(pl.LightningModule):
    def __init__(self, nstates, nactions):
        super().__init__()
        self.nstates = nstates
        self.nactions = nactions
        self.fc1 = torch.nn.Linear(self.nstates, self.nstates)
        self.fc2 = torch.nn.Linear(self.nstates, self.nactions * 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.Unflatten(1, (2, self.nactions))(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = utils.bsa_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class DataSet(torch.utils.data.Dataset):
    def __init__(self, belief_state, action_value, action_count):
        self.belief_state = torch.from_numpy(belief_state).float()
        self.action_value = torch.from_numpy(action_value).float()
        self.action_count = torch.from_numpy(action_count).float()

    def __getitem__(self, i):
        return self.belief_state[i], (self.action_value[i], self.action_count[i])

    def __len__(self):
        return len(self.belief_state)


train_loader = torch.utils.data.DataLoader(DataSet(bs, w_bsa))
model = NN(NSTATES, NACTIONS)
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_dataloaders=train_loader)
