# After downloading the books from Project Gutenberg, this script formats the books into a single file
#   and removes all non-letter characters, including spaces and newlines.
import re
from os import listdir
from os.path import join
from tqdm import tqdm
from string import ascii_lowercase


src = r"../data/books"
dst = r"../data/formatted_books.txt"
valid_chars = f"[^{ascii_lowercase}]"
start_token = r"\*\*\* START OF THE PROJECT GUTENBERG .+ \*\*\*"
end_token = r"\*\*\* END OF THE PROJECT GUTENBERG .+ \*\*\*"
## everything between the start/end tokens is the book

files = listdir(src)
# files = ["11.txt"] ## testing

with open(dst, "w", encoding="utf-8") as f_out:
    for file in tqdm(files):
        with open(join(src, file), "r", encoding="utf+8") as f_in:
            text = f_in.read()
            start_match = re.search(start_token, text) 
            end_match = re.search(end_token, text)
            if start_match is None or end_match is None: ## skip books without start/end tokens
                print(file)
                continue

            text = text[start_match.span()[1]:end_match.span()[0]].lower()
            formatted_text = re.sub(valid_chars, "", text) ## removes any non-letter chars
            f_out.write(formatted_text)


"""
excluded
40.txt
50.txt
52.txt
65.txt
"""