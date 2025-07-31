import torch

import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 3) En el loop de entrenamiento y evaluación, mover cada batch
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        # mover batch a MPS/CPU
        Xb = Xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        y_pred = model(Xb)
        loss   = loss_fn(y_pred, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    ys, yps = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yp = model(Xb)
            # traer a CPU antes de .numpy()
            ys.append(yb.cpu().numpy())
            yps.append(yp.cpu().numpy())
    y_true = np.vstack(ys)
    y_pred = np.vstack(yps)
    # desescalar
    # y_true_inv = scaler_y.inverse_transform(y_true)
    # y_pred_inv = scaler_y.inverse_transform(y_pred)
    return {
        'MAE':   mean_absolute_error(y_true, y_pred),
        'RMSE':  np.sqrt(mean_squared_error(y_true, y_pred)),
        'y_true': y_true,
        'y_pred': y_pred,
    }

def fit_model(model, train_loader, test_loader, device, epochs=200, lr=1e-3):
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    history = {'train_loss':[]}
    for ep in range(1, epochs+1):
        trl = train_epoch(model, train_loader, opt, loss_fn, device)
        history['train_loss'].append(trl)
        if ep % 5 == 0:
            print(f"Ep {ep}/{epochs} — Loss: {trl:.4f}")
    metrics = eval_model(model, test_loader, device)
    return history, metrics