import os
from translate_common import translation, detokenize_translation
from translate_from_transliteration import translate_transliteration_base


def translate_transliteration(sentence):
    tmp_file = "transliteration.tmp"
    with open(tmp_file, "w") as f:
        f.write(sentence)

    raw_result = translate_transliteration_base(tmp_file, True).stdout
    os.remove(tmp_file)

    output = ""
    for line in raw_result.decode().split('\n'):
        if translation(line):
            output += detokenize_translation(line, "NMT_input/tokenization/translation_bpe.model")

    return output


if __name__ == '__main__':
    sentence = input("Please enter a transliteration sentence for translation\n")
    print(translate_transliteration(sentence))
