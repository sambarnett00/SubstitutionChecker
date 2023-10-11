# script to calculate the frequency of each letter in the substituted books
from string import ascii_lowercase
from dotenv import load_dotenv
from os import environ

load_dotenv(".env")
src = environ["ENG_PATH"] ## path to 
chunksize = 4096

counts = {s: 0 for s in ascii_lowercase}
with open(src, "r", encoding="utf-8") as f_in:
    for let in f_in.read():
        counts[let] += 1

total = sum(counts.values())
freqs = {let: count / total for let, count in counts.items()}
out = sorted(ascii_lowercase, key=lambda x: freqs[x], reverse=True)

print(out)
print(freqs)

## ['e', 't', 'a', 'o', 'n', 'i', 's', 'r', 'h', 'd', 'l', 'u', 'c', 'm', 'f', 'g', 'w', 'p', 'y', 'b', 'v', 'k', 'j', 'x', 'z', 'q']