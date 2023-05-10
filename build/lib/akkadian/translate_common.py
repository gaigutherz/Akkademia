import sentencepiece
from translation_tokenize import TOKEN_DIR

def source(line):
    if len(line) < 1:
        return False

    if line[0] == 'S':
        return True

    return False


def translation(line):
    if len(line) < 1:
        return False

    if line[0] == 'D':
        return True

    return False


def detokenize_cuneiform(line, sp_dir):
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(sp_dir)

    splitted = line.split('\t')

    tokenized = ' '.join(splitted[1:])
    source = sp.decode_pieces(tokenized.split())
    return splitted[0] + ' ' + source.replace(' ','')


def detokenize_transliteration(line, sp_dir):
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(sp_dir)

    splitted = line.split('\t')

    tokenized = ' '.join(splitted[1:])
    source = sp.decode_pieces(tokenized.split())
    return splitted[0] + ' ' + source


def detokenize_translation(line, sp_dir, include_line_number=False):
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(sp_dir)

    splitted = line.split('\t')

    if len(splitted) > 1 :
        del splitted[1]

    tokenized = ' '.join(splitted[1:])
    translation = sp.decode_pieces(tokenized.split())
    if include_line_number:
        return splitted[0] + ' ' + translation
    return translation
