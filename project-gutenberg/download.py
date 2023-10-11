# Downloads the first 100 books from Project Gutenberg
import requests as r
from tqdm import tqdm
from os import makedirs
from os.path import exists

dst = r"../data/books"
if not exists(dst):
    makedirs(dst)

url_base = "https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt"

for i in tqdm(range(1, 100)):
    response = r.get(url_base.format(i))
    with open(f"{dst}/{i}.txt", "w", encoding="utf-8") as f_out:
        f_out.write(response.text)
