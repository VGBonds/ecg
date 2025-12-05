from torch.utils.data import TensorDataset, DataLoader
import torch

def get_dataloaders(train_emb, train_lbl, val_emb, val_lbl, test_emb=None, test_lbl=None,
                    batch_size=128, num_workers=0):

    pin_memory = True if torch.cuda.is_available() else False
    train_ds = TensorDataset(train_emb, train_lbl)
    val_ds   = TensorDataset(val_emb,   val_lbl)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    loaders = {"train": train_loader, "val": val_loader}

    if test_emb is not None and test_lbl is not None:
        test_ds = TensorDataset(test_emb, test_lbl)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        loaders["test"] = test_loader

    return loaders