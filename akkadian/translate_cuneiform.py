import os
from translate_common import translation, detokenize_translation
from translate_from_cuneiform import translate_cuneiform_base


def translate_cuneiform(sentence):
    with open("cuneiform.tmp", "w") as f:
        f.write(sentence)
        raw_result = translate_cuneiform_base(f, True).stdout

    os.remove("cuneiform.tmp")

    output = ""
    for line in raw_result.decode().split('\n'):
        if translation(line):
            output += detokenize_translation(line)

    return output


if __name__ == '__main__':
    sentence = input("Please enter a cuneiform sentence for translation\n")
    translate_cuneiform(sentence)
