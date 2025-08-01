import torch

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


def fit_model(model, train_loader, test_loader, device, epochs=200, lr=1e-3):
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    history = {'train_loss':[]}
    for ep in range(1, epochs+1):
        trl = train_epoch(model, train_loader, opt, loss_fn, device)
        history['train_loss'].append(trl)
        if ep % 5 == 0:
            print(f"Ep {ep}/{epochs} — Loss: {trl:.4f}")
    return history