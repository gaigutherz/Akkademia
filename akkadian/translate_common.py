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


def detokenize_english(line):
    splitted = line.split(' ', '\t')

    if len(splitted) > 1 :
        del splitted[1]

    tokenized = ' '.join(splitted)
    return tokenized.replace(' ▁', ' ').replace(' ,', ',').replace(' .', '.').replace(' -', '-')
