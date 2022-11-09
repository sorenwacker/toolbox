import pandas as pd
import numpy as np
import seaborn as sns

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from matplotlib.pyplot import scatter, title, show


print(f"CUDA is available: {torch.cuda.is_available()}")
print(
    f"CUDA current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name()}"
)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=2,
        layers=[64, 32],
        add_sigmoid=True,
        dropout=0,
        device=None,
        verbose=False,
        save_every=None,
        show_plots=False,
        random_state=None,
    ):
        """
        PyTorch based autoencoder providing a scikit-learn api.
        -----
        Args:
            - input_dim: int, number of input and output features
                for the neural network.
            - latent_dim: int, dimension of the latent or encoded
                space.
            - layers: Array(int), defines the architecture of the
                hidden layers.
            - add_sigmoid: bool, whether or not to add a sigmoida
                transformation
                at the middle layer
            - dropout: float, (0-1), dropout factor before each
                in the encoder
            - device: CUDA device to use
        """

        if random_state is not None:
            torch.manual_seed(random_state)

        super(AutoEncoder, self).__init__()

        self.save_every = save_every
        self.show_plots = show_plots
        self.snapshots = []

        layout = [input_dim] + layers + [latent_dim]
        layout = [(i, j) for i, j in zip(layout[:-1], layout[1:])]

        encoder = []

        for i, j in layout:
            encoder.append(nn.Dropout(dropout))
            encoder.append(nn.Linear(i, j))
            encoder.append(nn.ReLU())
        encoder.pop()

        if add_sigmoid:
            encoder.append(nn.Sigmoid())

        layout.reverse()

        decoder = []
        for j, i in layout:
            decoder.append(nn.Linear(i, j))
            decoder.append(nn.ReLU())
        decoder.pop()

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.snapshots = []
        self.device = device
        self.to(device)
        self._epoch = 0

        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)

        if verbose:
            for i in encoder:
                print(i)
            for i in decoder:
                print(i)
            if torch.cuda.is_available():
                print("Cuda available")
            else:
                print("Cuda is not available")
            print(f"Using device {current_device} ({current_device_name})")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(
        self,
        X_train,
        epochs,
        batch_size=8,
        labels=None,
        num_workers=4,
        shuffle=True,
        hue=None,
        lr=1e-4,
    ):
        """
        Fitting function to train the neural network.
        """

        if isinstance(X_train, pd.DataFrame):
            ndx = X_train.index
            X_train = X_train.values
            X_train = torch.tensor(X_train).float()

        dataloader = DataLoader(
            X_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        optimizer = optim.Adam(self.parameters(), lr=lr)

        criterion = nn.MSELoss()

        for i in tqdm(range(1, epochs + 1)):
            self._epoch += 1
            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                # compute reconstructions
                outputs = self(data)
                # compute training reconstruction loss
                train_loss = criterion(outputs, data)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # compute the epoch training loss
                assert train_loss is not np.NaN

            if (self.save_every is not None) and ((i) % self.save_every == 0):
                result = pd.DataFrame(
                    self.encoder(X_train.to(self.device)).detach().cpu().numpy(),
                    index=ndx,
                )

                result.columns = [c + 1 for c in result.columns]
                result = result.add_prefix("AE-")
                result["Epoch"] = self._epoch
                result["Labels"] = labels
                self.snapshots.append(result)
                if self.show_plots:
                    sns.relplot(
                        data=result,
                        x="AE-0",
                        y="AE-1",
                        hue="Labels",
                        kind="scatter",
                        height=3,
                    )
                    title(f"Epoch {i}, loss={train_loss:2.2f}")
                    show()

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            ndx = X.index
            X = X.values
        else:
            ndx = None
        X = torch.tensor(X).float().to(self.device)
        enc = self.encoder(X)
        enc = pd.DataFrame(enc)
        enc.columns = [c + 1 for c in enc.columns]
        enc = enc.add_prefix("AE-").astype(float)
        if ndx is not None:
            enc.index = ndx
        return enc
