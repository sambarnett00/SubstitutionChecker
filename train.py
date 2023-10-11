import torch
from torch.amp import autocast
from tqdm import tqdm
from os import makedirs, environ
from os.path import join, exists
from dotenv import load_dotenv
from dataloader import DataLoader
from model import Config, Transformer


class LRScheduler:
    """Linear warmup and inverse square root decay."""
    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 peak_lr: float,
                 current_step: int = 0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.current_step = current_step

    def step(self):
        self.current_step += 1
        lr = self.calculate_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_learning_rate(self):
        if self.current_step < self.warmup_steps:
            return self.peak_lr * self.current_step / self.warmup_steps
        else:
            return self.peak_lr / (self.current_step ** 0.5)


def main():
    torch.manual_seed(1337) ## karpathy's seed
    load_dotenv(".env")
    config = Config(
        vocab_size=26,
        n_classes=2,
        batch_size=64,
        context_length=64,
        d_model=384,
        n_head=6,
        n_layer=3,
        dropout=0.2,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    max_iter = 10000
    ckpt_period = 512
    ckpt_path = environ["CKPT_PATH"]
    
    transformer = Transformer(config).to(config.device)
    transformer.train()
    print(f"Model #params {transformer.n_params/1e6:.3f}M")
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    lr_scheduler = LRScheduler(optimizer, warmup_steps=512, peak_lr=1e-4)
    dl = DataLoader(environ["ENG_PATH"], environ["SUB_PATH"],
                    config.batch_size, config.context_length, config.device)
    
    ctx = autocast(device_type="cuda", dtype=torch.float16)
    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    prog_bar = tqdm(dl, total=max_iter)
    for i, (xb, yb) in enumerate(prog_bar, start=1):
        with ctx:
            _, loss = transformer(xb, yb)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        lossf = loss.item()
        prog_bar.set_description(f"loss: {lossf:.4f}")
        
        if i % ckpt_period == 0: ## save checkpoint
            checkpoint = {
                "model": transformer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_step": lr_scheduler.current_step,
            }
            if not exists(ckpt_path):
                makedirs(ckpt_path)
            torch.save(checkpoint, join(ckpt_path, f"train{i}.pt"))
        
        if i > max_iter:
            break



if __name__ == "__main__":
    main()
