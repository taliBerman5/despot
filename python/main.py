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
        loss = F.mse_loss(self(x), y)  # utils.bsa_loss(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        self.log("loss", loss)
        return {"loss": loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        loss = F.mse_loss(self(x), y)  # utils.bsa_loss(self(x), y)
        tensorboard_logs = {'test_loss': loss}
        self.log("test_loss", loss)
        return {"test_loss": loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_loss}
        return {"avg_loss": avg_loss, 'log': tensorboard_logs}

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


def learning_loop_over_step(csv_name, batch, epoch):
    belief_state, action_value, action_count, step = utils.load_csv(csv_name, NSTATES)
    max_step = int(max(step)[0])
    model = NN(NSTATES, NACTIONS)

    for i in range(1, max_step):
        trainer = pl.Trainer(max_epochs=epoch)
        print(f'starting step {i}')
        indices = np.argwhere(step.reshape(len(step)) <= i)
        belief_state_step, action_value_step, action_count_step = utils.step_values(indices, belief_state, action_value,
                                                                                    action_count)

        test_indices = np.argwhere(step.reshape(len(step)) == i + 1)
        belief_state_test, action_value_test, action_count_test = utils.step_values(test_indices, belief_state,
                                                                                    action_value,
                                                                                    action_count)

        train_loader = torch.utils.data.DataLoader(DataSet(belief_state_step, action_value_step, action_count_step),
                                                   batch_size=batch,
                                                   shuffle=True,
                                                   num_workers=8)
        test_loader = torch.utils.data.DataLoader(DataSet(belief_state_test, action_value_test, action_count_test),
                                                  batch_size=len(belief_state_test),
                                                  shuffle=False,
                                                  num_workers=8)

        trainer.fit(model, train_dataloaders=train_loader)
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == '__main__':
    utils.set_seed(1)
    learning_loop_over_step("pomcp_belief_statistics_per_action.csv", batch=16, epoch=2)
