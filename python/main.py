import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

NSTATES = 256
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
        MSE = torch.nn.MSELoss()
        loss = MSE(self(x), y)  # utils.bsa_loss(self(x), y)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        MSE = torch.nn.MSELoss()
        loss = MSE(self(x), y)  # utils.bsa_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class DataSet(torch.utils.data.Dataset):
    def __init__(self, belief_state, action_value, action_count):
        self.belief_state = torch.from_numpy(belief_state).float()
        self.action_value = torch.from_numpy(action_value).float()
        self.action_count = torch.from_numpy(action_count).float()

    def __getitem__(self, i):
        return self.belief_state[i], torch.stack((self.action_value[i], self.action_count[i]))

    def __len__(self):
        return len(self.belief_state)


def learning_loop_over_step(csv_name, epoch):
    belief_state, action_value, action_count, step = utils.load_csv(csv_name, NSTATES)
    max_step = int(max(step)[0])
    model = NN(NSTATES, NACTIONS)


    for i in range(1, max_step):
        trainer = pl.Trainer(max_epochs=epoch)
        print(f'starting step {i}')
        indices = np.argwhere(step.reshape(len(step)) <= i)
        belief_state_step = belief_state[indices].reshape(len(indices), belief_state.shape[1])
        action_value_step = action_value[indices].reshape(len(indices), action_value.shape[1])
        action_count_step = action_count[indices].reshape(len(indices), action_count.shape[1])

        test_indices = np.argwhere(step.reshape(len(step)) > i)
        belief_state_test = belief_state[test_indices].reshape(len(test_indices), belief_state.shape[1])
        action_value_test = action_value[test_indices].reshape(len(test_indices), action_value.shape[1])
        action_count_test = action_count[test_indices].reshape(len(test_indices), action_count.shape[1])

        train_loader = torch.utils.data.DataLoader(DataSet(belief_state_step, action_value_step, action_count_step), batch_size=16,
                                               shuffle=True)
        test_loader = torch.utils.data.DataLoader(DataSet(belief_state_test, action_value_test, action_count_test), batch_size=16)

        trainer.fit(model, train_dataloaders=train_loader)
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")


utils.set_seed(1)
learning_loop_over_step("pomcp_belief_statistics_per_action.csv", epoch=2)
