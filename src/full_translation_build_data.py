from build_data import build_signs_and_transcriptions, break_into_sentences, write_data_to_file
from pathlib import Path


def write_data_to_file(chars_sentences, translation_sentences):
    signs_file = open(Path(r"../NMT_input/signs_per_line.txt"), "w", encoding="utf8")
    transcription_file = open(Path(r"../NMT_input/transcriptions_per_line.txt"), "w", encoding="utf8")
    translation_file = open(Path(r"../NMT_input/translation_per_line.txt"), "w", encoding="utf8")

    for key in translation_sentences:
        for c in chars_sentences[key]:
            signs_file.write(c[3])
            delim = c[2] if not c[2] is None else " "
            transcription_file.write(c[1] + delim)

        for t in translation_sentences[key]:
            translation_file.write(t[1] + " ")

        signs_file.write("\n")
        transcription_file.write("\n")
        translation_file.write("\n")

    signs_file.close()
    transcription_file.close()
    translation_file.close()


def preprocess():
    chars, translation = build_signs_and_transcriptions(["rinap", "riao", "ribo"])
    chars_sentences = break_into_sentences(chars)
    translation_sentences = break_into_sentences(translation)
    write_data_to_file(chars_sentences, translation_sentences)


def main():
    preprocess()


if __name__ == '__main__':
    main()
