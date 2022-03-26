def source_or_translation(line):
    if len(line) < 1:
        return False

    if line[0] == 'S' or line[0] == 'D':
        return True

    return False
