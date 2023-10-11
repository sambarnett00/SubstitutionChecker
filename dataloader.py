import torch
from dotenv import load_dotenv
from os import environ


class DataLoader:
    """Custom DataLoader class for the Substitution Checker model
    A DataLoader object can be iterated indefinately through to get batches of data.
    Each batch of data is half english and half substituted text.
    Args:
        eng_src (str): path to the english formatted text
        sub_str (str): path to the substituted formatted text
        batch_size (int): batch size
        context_length (int): context length
        device (torch.device): device to load the data into
        training (bool): whether to load the training or test set. Defaults to True.
        train_test_split (float): train/test split. Defaults to 0.9.
    """
    def __init__(self, eng_src: str, sub_str: str,
                 batch_size: int, context_length: int, device: torch.device,
                 training: bool = True, train_test_split: float = 0.9) -> None:

        assert batch_size % 2 == 0, "batch size must be even"
        self._batch_size = batch_size
        self._context_length = context_length
        self._device = device
        self._training = training
        self._train_test_split = train_test_split

        eng_raw, sub_raw = DataLoader.read_raw(eng_src, sub_str)
        eng_split, sub_split = DataLoader.split(eng_raw, sub_raw, training, train_test_split)
        self._length = len(eng_split)
        self._eng, self._sub = DataLoader.tokenize(eng_split, sub_split)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a random batch of data"""
        ix = torch.randint(0, self._length-self._context_length, (self._batch_size//2,))
        batch_x = torch.stack([self._eng[i:i+self._context_length] for i in ix] + [self._sub[i:i+self._context_length] for i in ix])
        batch_y = torch.zeros(self._batch_size, dtype=torch.long)
        batch_y[self._batch_size//2:] = 1

        return batch_x.to(self._device), batch_y.to(self._device)

    @staticmethod
    def read_raw(eng_src: str, sub_src: str) -> tuple[str, str]:
        print("reading from disk...")
        with open(eng_src, 'r') as f_in:
            eng_raw = f_in.read()
        with open(sub_src, 'r') as f_in:
            sub_raw = f_in.read()
        return eng_raw, sub_raw
    
    @staticmethod
    def split(eng_raw: str, sub_raw: str, training: bool, train_test_split: float) -> tuple[str, str]:
        mid = int(len(eng_raw) * train_test_split)
        if training: return eng_raw[:mid], sub_raw[:mid]
        else: return eng_raw[mid:], sub_raw[mid:]
    
    @staticmethod
    def tokenize(eng_raw: str, sub_raw: str) -> tuple[torch.Tensor, torch.Tensor]:
        print("tokenizing...")
        eng = torch.tensor([ord(c) - 97 for c in eng_raw])
        sub = torch.tensor([ord(c) - 97 for c in sub_raw])
        return eng, sub
    
    @staticmethod
    def decode(tensor: torch.Tensor) -> str:
        return ''.join([chr(c + 97) for c in tensor.view(-1)])


if __name__ == "__main__":
    load_dotenv(".env")
    dl = DataLoader(environ["ENG_PATH"], environ["SUB_PATH"], 
                    32, 64, torch.device("cpu"))
    inp, target = next(dl)
    print(inp)
    print(dl.decode(inp))
    print(target)
