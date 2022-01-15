import subprocess
from pathlib import Path
from translation_tokenize import tokenize


def translate_from_transliteration(file):
    tokenize("transliteration_bpe", file, False, Path("NMT_input/tokenization"), Path(""), Path("/tmp"))
    cmd = "../fairseq/fairseq_cli/interactive.py " \
          "data-bin-transliteration/ " \
          "--path trans_result.LR_0.1.MAX_TOKENS_4000/checkpoint_best.pt " \
          "--beam 5 " \
          "--input /tmp/" + file
    result = subprocess.run(cmd.split())
    print(result.stdout)


if __name__ == '__main__':
    transliteration_file = input("Please enter the name of the transliteration file for translation\n")
    translate_from_transliteration(transliteration_file)
