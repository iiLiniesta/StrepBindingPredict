import hashlib
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

import stall_utils_slim
from stall_utils_slim import nt_onehot_mapping, aa_onehot_mapping

# assert torch.cuda.is_available()
Device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else "cpu")
Logger = stall_utils_slim.create_logger(log_dir="./output/")


def seq2onehot(seqs: list[str], mapping: dict[str, int], verbose=True):
    if verbose:
        do_log = Logger.info
    else:
        do_log = Logger.debug
    onehots = []
    do_log("Converting one-hot ...")
    for seq in tqdm(seqs, desc="Converting one-hot ...", bar_format=stall_utils_slim.TqdmBarFormat):
        onehots.append(
            nn.functional.one_hot(
                torch.tensor([mapping[c] for c in seq]).to(torch.int64),
                num_classes=len(mapping),
            ).float().T
        )
    ans = torch.stack(onehots)
    return ans


class StallingRnnNet(nn.Module):  # Batch First: True.
    def __init__(
            self,
            seq_length: int,
            onehot_mapping_dict: dict,
            rnn_hidden_size: int = 64,
            rnn_num_layers: int = 8,
            mlp_hidden_size1: int = 128,
            mlp_hidden_size2: int = 32,
            # hidden_to_mlp: bool = True,
    ):
        Logger.info("Initializing StallingRnnNet ...")
        super(StallingRnnNet, self).__init__()

        # self.hidden_to_mlp = hidden_to_mlp
        # Logger.info("hidden_to_mlp = %s", hidden_to_mlp)
        self.seq_length = seq_length
        self.input_size = input_size = len(onehot_mapping_dict)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
            batch_first=True, bidirectional=False,
        )
        Logger.info(self.rnn)
        # d_mlp_in = 1 * rnn_hidden_size + (rnn_num_layers * rnn_hidden_size if hidden_to_mlp else 0)
        d_mlp_in = rnn_num_layers * rnn_hidden_size
        self.MLP = nn.Sequential(
            nn.Linear(d_mlp_in, mlp_hidden_size1, bias=True),
            # nn.LayerNorm
            nn.BatchNorm1d(mlp_hidden_size1, momentum=0.01),
            nn.ELU(),  # LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(mlp_hidden_size1, mlp_hidden_size2, bias=True),
            nn.BatchNorm1d(mlp_hidden_size2, momentum=0.01),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(mlp_hidden_size2, 1, bias=True),
        )
        Logger.info(self.MLP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x_target_shape = torch.Size([x.shape[0], self.seq_length, self.input_size])
        assert x.shape == x_target_shape, "x.shape = %s, instead of %s" % (x.shape, x_target_shape)

        y, h = self.rnn(x, None)
        # y.shape: (batch_size, sequence_length, rnn_hidden_size)
        # h.shape: (rnn_num_layers, batch_size, input_size)
        y_target_shape = torch.Size([y.shape[0], self.seq_length, self.rnn_hidden_size])
        h_target_shape = torch.Size([self.rnn_num_layers, h.shape[1], self.rnn_hidden_size])
        assert y.shape == y_target_shape, "y.shape = %s, instead of %s" % (y.shape, y_target_shape)
        assert h.shape == h_target_shape, "h.shape = %s, instead of %s" % (h.shape, h_target_shape)
        h = h.transpose(0, 1)

        # y = y[:, -1:, :]
        # mlp_input = torch.cat((y, h), dim=1) if self.hidden_to_mlp else y  # Concatenate and go through MLP.
        mlp_input = h
        mlp_input = mlp_input.reshape((mlp_input.shape[0], -1))
        # print(mlp_input.shape)
        mlp_out = self.MLP(mlp_input)
        return mlp_out


class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def predict_once_same_length(
        seqs: list[str], mode, model_fn: str, seq_length: int,
        batch_size: int, verbose=True,
) -> list[float]:
    # Length of every sequence in "seqs" must be "seq_length".
    mode = mode.upper()
    if mode == 'NT':
        onehot_mapping = nt_onehot_mapping
    elif mode == 'AA':
        onehot_mapping = aa_onehot_mapping
    else:
        assert False

    if verbose:
        do_log = Logger.info
    else:
        do_log = Logger.debug

    do_log("Loading model from '%s'" % model_fn)
    with open(model_fn, "rb") as f:
        bs = f.read()  # read file as bytes
        hash_md5 = hashlib.md5(bs).hexdigest()
        hash_sha256 = hashlib.sha256(bs).hexdigest()
        do_log("MD5: %s, SHA256: %s." % (hash_md5, hash_sha256))
    model = torch.load(model_fn, weights_only=False).cuda().to(Device)
    do_log("model.seq_length: %d" % model.seq_length)
    model.eval()

    bigger_batch_size = batch_size * 1024
    n = len(seqs)
    y_preds = []
    for i in range(0, n, bigger_batch_size):
        bigger_batch_seqs = seqs[i: i + bigger_batch_size]
        onehot_seqs = seq2onehot(bigger_batch_seqs, onehot_mapping, verbose=verbose)
        m = len(onehot_seqs)
        all_data = []
        do_log("Loading one-hot ...")
        for j in trange(m, desc="Loading one-hot:", bar_format=stall_utils_slim.TqdmBarFormat, ):
            all_data.append(onehot_seqs[j].cuda().to(Device))
        do_log("Predicting %d sequences..." % (m,))
        all_dataset = MyDataSet(all_data)
        all_loader = DataLoader(all_dataset, shuffle=False, batch_size=batch_size, drop_last=False)

        for j, data_x in enumerate(tqdm(
                all_loader,
                desc="Predicting", bar_format=stall_utils_slim.TqdmBarFormat,
        )):
            pred = model(data_x)
            y_preds += list(pred.cpu().squeeze(1).detach().numpy().tolist())
        do_log("Done prediction: %d / %d." % (len(y_preds), n))
    return y_preds


