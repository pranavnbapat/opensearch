from nltk.corpus import words
import nltk
nltk.download('words')

ENGLISH_VOCAB = set(w.lower() for w in words.words())
MIN_WORD_LEN = 3

def find_glued_words(text, vocab=ENGLISH_VOCAB, min_word_len=MIN_WORD_LEN):
    suspicious = []
    for token in text.split():
        t = token.lower()
        if t in vocab or not t.isalpha() or len(t) < 2 * min_word_len:
            continue
        # Try every split point
        for i in range(min_word_len, len(t) - min_word_len + 1):
            left, right = t[:i], t[i:]
            if left in vocab and right in vocab:
                suspicious.append((token, left, right))
                break
    return suspicious
