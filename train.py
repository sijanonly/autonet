import time

import numpy as np
import torch

class TrainManager:
    """ A helper class to train, evaluate the model"""

    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch

    @property
    def avg_validation_loss(self):
        val_loss_arr = np.array(self.val_losses)
        val_avg_loss = val_loss_arr.mean()
        if val_avg_loss < 1:
            val_loss = -np.log(val_avg_loss) + 1
        else:
            val_loss = np.exp(-(val_avg_loss - 1))

        return val_loss

    def train(self, train_dataset, val_dataset, batch_size=50, n_epochs=50):
        for epoch in range(n_epochs):
            ###################
            # train the model
            ###################
            self.model.train()
            # self.model.reset_hidden_state()
            # h = self.model.init_hidden(batch_size)
            train_loss = 0
            start_time = time.time()
            for x_batch, y_batch, batch in self.generate_batch_data(
                train_dataset.X, train_dataset.y, batch_size
            ):
                self.model.reset_hidden_state()
                outputs = self.model(x_batch)
                self.optimizer.zero_grad()
                tloss = self.criterion(outputs, y_batch)
                tloss.backward()
                self.optimizer.step()
                train_loss += tloss.item()
            elapsed = time.time() - start_time
            train_loss /= batch
            self.train_losses.append(train_loss)

            self._validation(val_dataset.X, val_dataset.y, batch_size)  
       
            print('training for {} epoch completed'.format(epoch))

    def _validation(self, x_val, y_val, batch_size):
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(
                x_val, y_val, batch_size
            ):
                outputs = self.model(x_batch)
                vloss = self.criterion(outputs, y_batch)
                val_loss += vloss.item()
            val_loss /= batch
            self.val_losses.append(val_loss)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(
                x_test, y_test, batch_size
            ):
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch) :]
                    if y_pred.shape[1] > y_batch.shape[1]
                    else y_pred
                )
                loss = self.criterion(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch
            return actual, predicted, test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
