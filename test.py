import torch
from torch.amp import autocast
from argparse import ArgumentParser
from os import environ
from os.path import join
from dotenv import load_dotenv
from tqdm import tqdm
from model import Transformer, Config
from dataloader import DataLoader


def test(filename: str, seed: int):
    torch.manual_seed(seed)
    load_dotenv(".env")

    ckpt_file = filename
    ckpt_path = environ["CKPT_PATH"]
    ckpt = torch.load(join(ckpt_path, ckpt_file))

    config: Config = ckpt["config"]
    transformer: Transformer = Transformer(config).to(config.device)
    transformer.load_state_dict(ckpt["model"])
    transformer.eval()
    
    dl = DataLoader(environ["ENG_PATH"], environ["SUB_PATH"],
                    config.batch_size, config.context_length, config.device,
                    training=False, train_test_split=0.0)
    ctx = autocast(device_type="cuda", dtype=torch.float16)
    
    test_iter = 512
    total = correct = 0
    prog_bar = tqdm(dl, total=test_iter)  
    for i, (xb, yb) in enumerate(prog_bar, start=1):
        with ctx:
            predictions = transformer.predict(xb)

        correct += (predictions == yb).sum().item()
        total += predictions.size(0)
        prog_bar.set_description(f"Acc: {correct / total:.4f}")
        
        if i > test_iter:
            break
    
    print(f"Transformer Model - #params: {transformer.n_params}")
    print(f"Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    argparser = ArgumentParser(prog="test.py",
                               description="Test Transformer model")
    argparser.add_argument("filename", type=str, help="checkpoint filename")
    argparser.add_argument("-s", "--seed", type=int, default=1337,
                           help="random seed for reproducibility")
    args = argparser.parse_args()
    test(args.filename, args.seed)
