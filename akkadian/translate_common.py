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