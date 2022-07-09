import os
from translate_common import translation, detokenize_translation
from translate_from_cuneiform import translate_cuneiform_base


def translate_cuneiform(sentence):
    tmp_file = "cuneiform.tmp"
    with open(tmp_file, "w") as f:
        f.write(sentence)

    raw_result = translate_cuneiform_base(tmp_file, True).stdout
    os.remove(tmp_file)

    output = ""
    for line in raw_result.decode().split('\n'):
        if translation(line):
            output += detokenize_translation(line,
                                             "NMT_input/not_divided_by_three_dots/tokenization/translation_bpe.model")

    return output


if __name__ == '__main__':
    sentence = input("Please enter a cuneiform sentence for translation\n")
    print(translate_cuneiform(sentence))
