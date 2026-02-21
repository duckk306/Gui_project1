class Trainer:
    def __init__(self, model, criterion, optimizer, device="cuda"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            imgs = batch["image"].to(self.device)
            targets = {
                "drive_area": batch["drive_area"].to(self.device),
                "detection": None
            }

            outputs = self.model(imgs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}"
                )

        return total_loss / len(loader)
