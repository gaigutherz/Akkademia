import subprocess
from pathlib import Path
from translation_tokenize import tokenize
from akkadian.translate_common import source_or_translation


def translate_from_cuneiform_base(file):
    tokenize("signs_char", file, False, Path("NMT_input/tokenization"), Path(""), Path("/tmp"))
    cmd = "../fairseq/fairseq_cli/interactive.py " \
          "data-bin-not-divided-by-three-dots/ " \
          "--path not_divided_by_three_dots_result.LR_0.1.MAX_TOKENS_4000/checkpoint_best.pt " \
          "--beam 5 " \
          "--input /tmp/" + file
    return subprocess.run(cmd.split())


def translate_from_cuneiform_raw(file):
    print(translate_from_cuneiform_base(file).stdout)


def translate_from_cuneiform(file):
    raw_result = translate_from_cuneiform_base(file).stdout
    for line in raw_result:
        if not source_or_translation(line):
            continue
        print(line)


if __name__ == '__main__':
    cuneiform_file = input("Please enter the name of the Akkadian file for translation\n")
    translate_from_cuneiform(cuneiform_file)
