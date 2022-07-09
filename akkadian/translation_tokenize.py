import sentencepiece
from pathlib import Path
import shutil

DIVIDED_BY_THREE_DOTS = False

if DIVIDED_BY_THREE_DOTS == True:
    BASE_DIR = Path("../NMT_input")
else:
    BASE_DIR = Path("../NMT_input") / Path("not_divided_by_three_dots")

TOKEN_DIR = BASE_DIR / Path("tokenization")

TRAIN_AK = Path("train.ak")
TRAIN_TR = Path("train.tr")
TRAIN_EN = Path("train.en")
VALID_AK = Path("valid.ak")
VALID_TR = Path("valid.tr")
VALID_EN = Path("valid.en")
TEST_AK = Path("test.ak")
TEST_TR = Path("test.tr")
TEST_EN = Path("test.en")
FOR_TRANSLATION_TR = Path("for_translation.tr")


def train_and_move(input_file, model_type, model_prefix, vocab_size):
    sentencepiece.SentencePieceTrainer.train(f'--input={input_file} --model_type={model_type} --model_prefix={model_prefix} --vocab_size={vocab_size}')

    f = model_prefix + ".model"
    shutil.move(f, TOKEN_DIR / f)
    f = model_prefix + ".vocab"
    shutil.move(f, TOKEN_DIR / f)


def train_tokenizer():
    train_and_move(BASE_DIR / TRAIN_AK, "char", "signs_char", 400)
    # train_and_move(BASE_DIR / TRAIN_AK, "bpe", "signs_bpe", 400)
    train_and_move(BASE_DIR / TRAIN_TR, "bpe", "transliteration_bpe", 1000)
    train_and_move(BASE_DIR / TRAIN_EN, "bpe", "translation_bpe", 10000)


def tokenize(model_prefix, file, should_remove_prefix=False, token_dir=TOKEN_DIR, base_dir=BASE_DIR, output_dir=TOKEN_DIR):
    sp = sentencepiece.SentencePieceProcessor()
    f = model_prefix + ".model"
    sp.load(str(token_dir / f))

    with open(base_dir / file, "r", encoding="utf8") as fin:
        data = fin.readlines()

    if should_remove_prefix:
        tokenized_data = [" ".join(sp.encode_as_pieces(line.split(": ", 1)[1])) for line in data]
    else:
        tokenized_data = [" ".join(sp.encode_as_pieces(line)) for line in data]
    #print('\n'.join(tokenized_data))

    output_file = output_dir / file
    with open(output_file, "w", encoding="utf8") as fout:
        for line in tokenized_data:
            fout.write(line + "\n")


def detokenize_atae_translated():
    sp1 = sentencepiece.SentencePieceProcessor()
    sp1.load(str(TOKEN_DIR / "transliteration_bpe.model"))

    sp2 = sentencepiece.SentencePieceProcessor()
    sp2.load(str(TOKEN_DIR / "translation_bpe.model"))

    with open(Path("../atae_translated.txt"), "r", encoding="utf8") as fin:
        data = fin.readlines()

    detokenized_data = []
    for line in data:
        if line[0] == 'S':
            parts = line.split("\t", 1)
            detokenized_data.append(parts[0] + "\t" + sp1.decode_pieces(parts[1].split(" ")))
        elif line[0] == 'H' or line[0] == 'D':
            parts = line.split("\t", 2)
            detokenized_data.append(parts[0] + "\t" + parts[1] + "\t" + sp2.decode_pieces(parts[2].split(" ")))
        else:
            detokenized_data.append(line)

    with open(Path("../atae_translated_detokenized.txt"), "w", encoding="utf8") as fout:
        for line in detokenized_data:
            fout.write(line)


def detokenize_best_run_test_data_translated(only_core_data):
    sp1 = sentencepiece.SentencePieceProcessor()
    sp1.load(str(TOKEN_DIR / "transliteration_bpe.model"))

    sp2 = sentencepiece.SentencePieceProcessor()
    sp2.load(str(TOKEN_DIR / "translation_bpe.model"))

    with open(Path("../best_run_test_data_translated.txt"), "r", encoding="utf8") as fin:
        data = fin.readlines()

    detokenized_data = []
    for line in data:
        if line[0] == 'S':
            if not only_core_data:
                parts = line.split("\t", 1)
                detokenized_data.append(parts[0] + "\t" + sp1.decode_pieces(parts[1].split(" ")).replace("_", " "))
        elif line[0] == 'T':
            parts = line.split("\t", 1)
            if only_core_data:
                detokenized_data.append("<gold>: " + sp2.decode_pieces(parts[1].split(" ")).replace("_", " "))
            else:
                detokenized_data.append(parts[0] + "\t" + sp2.decode_pieces(parts[1].split(" ")).replace("_", " "))
        elif line[0] == 'H' or line[0] == 'D':
            parts = line.split("\t", 2)
            if only_core_data:
                if line[0] == 'H':
                    detokenized_data.append("<predicted>: " + sp2.decode_pieces(parts[2].split(" ")).replace("_", " ") + "\n")
            else:
                detokenized_data.append(parts[0] + "\t" + parts[1] + "\t" + sp2.decode_pieces(parts[2].split(" ")).replace("_", " "))
        else:
            if not only_core_data:
                detokenized_data.append(line)

    with open(Path("../best_run_test_data_translated_detokenized.txt"), "w", encoding="utf8") as fout:
        for line in detokenized_data:
            fout.write(line)


def run_tokenizer():
    # TODO: Compare signs_chars to signs_bpe
    tokenize("signs_char", TRAIN_AK)
    tokenize("signs_char", VALID_AK)
    tokenize("signs_char", TEST_AK)

    tokenize("transliteration_bpe", TRAIN_TR)
    tokenize("transliteration_bpe", VALID_TR)
    tokenize("transliteration_bpe", TEST_TR)

    tokenize("translation_bpe", TRAIN_EN)
    tokenize("translation_bpe", VALID_EN)
    tokenize("translation_bpe", TEST_EN)


def tokenize_transliteration_for_translation():
    tokenize("transliteration_bpe", FOR_TRANSLATION_TR, True)


def main():
    train_tokenizer()
    run_tokenizer()
    # tokenize_transliteration_for_translation()
    # detokenize_atae_translated()
    # detokenize_best_run_test_data_translated(True)


if __name__ == '__main__':
    main()
