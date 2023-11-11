# Substitution Checker
In order to crack some substitution ciphers, we need a heuristic to determine how "decypted" a piece of text is.</br>
Introduce the substitution checker, a model with a encoder transformer architecture to evaluate a piece of text's "decyptedness".

## Problem background
As a part of an exercise in cryptography, I was tasked with decrypting some ciphertext which had been encrypted using a substitution cipher.</br>
I was previously scanning the decrypted plaintext and counting the number of english words present as a heuristic for the "decryptedness".</br>
I also experimented with adding a weight to the english word based on its frequency:
- More common words have a higher weight
- Less commmon words have a higher weight

I would then use this heuristic to evaluate different substitution keys, iteratively getting closer to the real key used.</br>
However, for plaintext with unusual letter frequencies, this heuristic was not affective and so I developed a machine learning approach.