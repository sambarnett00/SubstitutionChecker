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
        self._eng, self._sub = DataLoader.tokenize(eng_raw, sub_raw)
        self._length = self._eng.size(0)
        self._lower_bound, self._upper_bound = self._get_bounds()

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a batch of data"""
        ix = torch.randint(self._lower_bound, self._upper_bound, (self._batch_size//2,))
        batch_x = torch.stack([self._eng[i:i+self._context_length] for i in ix] + [self._sub[i:i+self._context_length] for i in ix])
        batch_y = torch.zeros(self._batch_size, dtype=torch.long)
        batch_y[self._batch_size//2:] = 1
        # batch_y[self._batch_size//2:, 1] = 1

        return batch_x.to(self._device), batch_y.to(self._device)

    
    def set_training(self, training: bool) -> None:
        self._training = training
        self._lower_bound, self._upper_bound = self._get_bounds()
    
    def _get_bounds(self):
        """Calculates the lower and upper bounds for the data to be used when creating a batch"""
        if self._training:  # 0 -> 0.9
            return 0, int(self._length * self._train_test_split - self._context_length)
        else:               # 0.9 -> 1
            return int(self._length * self._train_test_split), int(self._length - self._context_length)

    @staticmethod
    def read_raw(eng_src: str, sub_src: str) -> tuple[str, str]:
        print("reading from disk...")
        with open(eng_src, 'r') as f_in:
            eng_raw = f_in.read()
        with open(sub_src, 'r') as f_in:
            sub_raw = f_in.read()
        return eng_raw, sub_raw
    
    @staticmethod
    def tokenize(eng_raw: str, sub_raw: str) -> tuple[torch.Tensor, torch.Tensor]:
        print("tokenizing...")
        eng, sub = torch.zeros(len(eng_raw), dtype=torch.long), torch.zeros(len(sub_raw), dtype=torch.long)
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
