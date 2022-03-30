import subprocess
from pathlib import Path
from translation_tokenize import tokenize
from translate_common import source, translation, detokenize_source, detokenize_translation


def translate_cuneiform_base(file, capture_output=False):
    tokenize("signs_char", file, False, Path("NMT_input/tokenization"), Path(""), Path("/tmp"))
    cmd = "../fairseq/fairseq_cli/interactive.py " \
          "data-bin-not-divided-by-three-dots/ " \
          "--path not_divided_by_three_dots_result.LR_0.1.MAX_TOKENS_4000/checkpoint_best.pt " \
          "--beam 5 " \
          "--input /tmp/" + file

    if capture_output:
        return subprocess.run(cmd.split(), capture_output=True)
    return subprocess.run(cmd.split())


def translate_cuneiform_raw(file):
    translate_cuneiform_base(file)


def translate_cuneiform_file(file):
    raw_result = translate_cuneiform_base(file, True).stdout
    for line in raw_result.decode().split('\n'):
        if source(line):
            print(detokenize_source(line))
        if translation(line):
            print(detokenize_translation(line))


def translate_cuneiform(sentence):
    with open("cuneiform.tmp", encoding='utf-8') as f:
        f.write(sentence)
        raw_result = translate_cuneiform_base(f, True).stdout

    for line in raw_result.decode().split('\n'):
        if translation(line):
            print(detokenize_translation(line))


if __name__ == '__main__':
    cuneiform_file = input("Please enter the name of the Akkadian file for translation\n")
    translate_cuneiform_file(cuneiform_file)
