import torch
from torch.cuda.amp import autocast
from os import environ
from os.path import join
from dotenv import load_dotenv
from tqdm import tqdm
from model import Transformer, Config
from dataloader import DataLoader


def main():
    torch.manual_seed(1337)
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

    dl = DataLoader(environ["ENG_PATH"], environ["SUB_PATH"],
                    config.batch_size, config.context_length, config.device,
                    training=False)

    ckpt_file = "train10240.pt"
    ckpt_path = environ["CKPT_PATH"]
    ckpt = torch.load(join(ckpt_path, ckpt_file))
    transformer: Transformer = ckpt["transformer"].to(config.device)
    transformer.eval()
    
    ctx = autocast(device_type="cuda", dtype=torch.float16)

    test_iter = 512
    prog_bar = tqdm(dl, total=test_iter)    
    for i, (xb, yb) in enumerate(prog_bar, start=1):
        with torch.no_grad(), ctx:
            logits, _ = transformer(xb, yb)
        lossf = loss.item()
        prog_bar.set_description(f"loss: {lossf:.4f}")
        
        if i > test_iter:
            break

if __name__ == "__main__":
    main()
