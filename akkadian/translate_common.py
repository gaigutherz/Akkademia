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


def detokenize_cuneiform(line):
    splitted = line.split('\t')

    tokenized = ' '.join(splitted[1:])
    source = tokenized.replace('▁', '').replace(' ', '')
    return splitted[0] + ' ' + source


def detokenize_transliteration(line):
    splitted = line.split('\t')

    tokenized = ' '.join(splitted[1:])
    source = tokenized.replace('▁', '').replace('- ', '-').replace(' -', '-').replace('. ', '.').replace(' .', '.').replace('{ ', '{').replace(' }', '}')
    return splitted[0] + ' ' + source


def detokenize_translation(line, include_line_number=False):
    splitted = line.split('\t')

    if len(splitted) > 1 :
        del splitted[1]

    tokenized = ' '.join(splitted[1:])
    translation = tokenized.replace('▁', '').replace(' ,', ',').replace(' .', '.').replace('- ', '-').replace(' -', '-').replace(' !', '!').replace(' ?', '?').replace(' ;', ';').replace(' \'', '\'').replace('\' ', '\'').replace(' ʾ', 'ʾ').replace('ʾ ', 'ʾ').replace('( ', '(').replace(' )', ')')
    if include_line_number:
        return splitted[0] + ' ' + translation
    return translation

def detokenize_translation_using_sp(line, base_dir, include_line_number=False):
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(str(base_dir / "tokenization/translation_bpe.model"))

    splitted = line.split('\t')

    if len(splitted) > 1 :
        del splitted[1]

    translation = sp.decode_pieces(splitted[1:])
    if include_line_number:
        return splitted[0] + ' ' + translation
    return translation
