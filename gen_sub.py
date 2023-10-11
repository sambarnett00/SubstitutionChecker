# This script generates a substituted version of the books
# chunks of 4096 characters are substituted with unique substitution ciphers
from string import ascii_lowercase
from random import shuffle
from tqdm import tqdm
from dotenv import load_dotenv
from os import environ


def substitute(text, key):
    return "".join([key[ascii_lowercase.index(let)] for let in text])


load_dotenv()
src = environ["SRC_PATH"] ## 
alphabet = list(ascii_lowercase)
chunksize = 4096

with open(src, "r", encoding="utf-8") as f_in:
    text = f_in.read()

out = ""
l = len(text)
for i in tqdm(range(l // chunksize + 1)):
    key = alphabet.copy()
    shuffle(key)

    text_chunk = text[i * chunksize: min((i + 1) * chunksize, l)]
    out += substitute(text_chunk, key)


with open(environ["SUB_PATH"], "w", encoding="utf-8") as f_out:
    f_out.write(out)
