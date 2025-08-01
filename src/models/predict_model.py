import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def eval_model(model, loader, device):
    model.eval()
    ys, yps = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yp = model(Xb)
            ys.append(yb.cpu().numpy())
            yps.append(yp.cpu().numpy())
    y_true = np.vstack(ys)
    y_pred = np.vstack(yps)
    return {
        'MAE':   mean_absolute_error(y_true, y_pred),
        'RMSE':  np.sqrt(mean_squared_error(y_true, y_pred)),
        'y_true': y_true,
        'y_pred': y_pred,
    }
