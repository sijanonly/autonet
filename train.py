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

        self.indices = None
    
    def shuffle_on_epoch(self):
        np.random.shuffle(self.indices)
    
    @classmethod
    def generate_batch_data(cls, x, y, batch_size, is_validation=False):

        if cls.indices.size == 0:
            cls.indices = np.arange(x.shape[0])

        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            if is_validation:
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
            else:
                inds = cls.indices[i : i + batch_size]
                x_batch = x[inds]
                y_batch = y[inds]
            yield x_batch, y_batch, batch  
            
    @property
    def avg_validation_loss(self):
        val_loss_arr = np.array(self.val_losses)
        val_avg_loss = val_loss_arr.mean()
        return val_avg_loss

    def train(self, train_dataset, val_dataset, batch_size=10, n_epochs=20, is_shuffle=False):

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
            if is_shuffle:
                self.shuffle_on_epoch()

    def _validation(self, x_val, y_val, batch_size):
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(
                x_val, y_val, batch_size,is_validation=True
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
