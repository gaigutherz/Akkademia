import sentencepiece
from pathlib import Path
import shutil


SIGNS = Path("../NMT_input/signs.txt")
TRANSLITERATION = Path("../NMT_input/transcriptions.txt")
TRANSLATION = Path("../NMT_input/translation.txt")


def train_and_move(input_file, model_type, model_prefix, vocab_size):
    sentencepiece.SentencePieceTrainer.train(f'--input={input_file} --model_type={model_type} --model_prefix={model_prefix} --vocab_size={vocab_size}')

    f = model_prefix + ".model"
    shutil.move(f, Path("../NMT_input/tokenization") / f)
    f = model_prefix + ".vocab"
    shutil.move(f, Path("../NMT_input/tokenization") / f)


def train_tokenizer():
    train_and_move(SIGNS, "char", "signs_char", 400)
    train_and_move(SIGNS, "bpe", "signs_bpe", 400)
    train_and_move(TRANSLITERATION, "bpe", "transliteration_bpe", 1000)
    train_and_move(TRANSLATION, "bpe", "translation_bpe", 10000)


def tokenize(model_prefix, file, n):
    sp = sentencepiece.SentencePieceProcessor()
    f = model_prefix + ".model"
    sp.load(str(Path("../NMT_input/tokenization") / f))

    with open(file, "r", encoding="utf8") as fin:
        data = [line for line in fin][:n]

    tokenized_data = [" ".join(sp.encode_as_pieces(line)) for line in data]
    print('\n'.join(tokenized_data))


def run_tokenizer():
    tokenize("signs_char", SIGNS, 5)
    tokenize("signs_bpe", SIGNS, 5)
    tokenize("transliteration_bpe", TRANSLITERATION, 5)
    tokenize("translation_bpe", TRANSLATION, 5)


def main():
    train_tokenizer()
    run_tokenizer()


if __name__ == '__main__':
    main()
