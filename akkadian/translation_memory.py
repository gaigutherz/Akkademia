from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu
from statistics import mean

# DIVIDED_BY_THREE_DOTS works well for T2E, and worst for C2E.
DIVIDED_BY_THREE_DOTS = False

if DIVIDED_BY_THREE_DOTS == True:
    BASE_DIR = Path("../NMT_input") / Path("tokenization")
    TRAIN_INPUT = BASE_DIR / Path("train.tr")
    TEST_INPUT = BASE_DIR / Path("test.tr")
    TRANSLATION_OUTPUT = BASE_DIR / Path("t2e_translation_memory.en")
else:
    BASE_DIR = Path("../NMT_input") / Path("not_divided_by_three_dots") / Path("tokenization")
    TRAIN_INPUT = BASE_DIR / Path("train.ak")
    TEST_INPUT = BASE_DIR / Path("test.ak")
    TRANSLATION_OUTPUT = BASE_DIR / Path("c2e_translation_memory_2.en")

TRAIN_OUTPUT = BASE_DIR / Path("train.en")
TEST_OUTPUT = BASE_DIR / Path("test.en")

translation_memory_dict = {}


def build_translation_memory_dict():
    with open(TRAIN_INPUT, "r", encoding="utf8") as fin, open(TRAIN_OUTPUT, "r", encoding="utf8") as fout:
        for lin, lout in zip(fin, fout):
            translation_memory_dict[lin] = lout


def translate(line):
    best_bleu_score = 0
    best_translation = ""
    for candidate in translation_memory_dict:
        blue_score = sentence_bleu([line.split()], candidate.split())
        if blue_score > best_bleu_score:
            print(blue_score * 100)
            best_translation = translation_memory_dict[candidate]
            best_bleu_score = blue_score
    return best_translation


def translation_memory_translate():
    with open(TEST_INPUT, "r", encoding="utf8") as fin, open(TRANSLATION_OUTPUT, "w", encoding="utf8") as fout:
        i = 1
        for lin in fin:
            print("Writing to file translation number " + str(i))
            fout.write(translate(lin))
            i += 1


def translation_memory_compute_bleu():
    bleu_scores = []
    with open(TEST_OUTPUT, "r", encoding="utf8") as fgold, open(TRANSLATION_OUTPUT, "r", encoding="utf8") as fpredict:
        for lgold, lpredict in zip(fgold, fpredict):
            bleu_score = sentence_bleu([lgold.split()], lpredict.split())
            bleu_scores.append(bleu_score)

    print(mean(bleu_scores) * 100)


if __name__ == '__main__':
    build_translation_memory_dict()
    translation_memory_translate()
    translation_memory_compute_bleu()
