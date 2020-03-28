import xml.etree.ElementTree as ET


def from_key_to_line_number(k):
    n = k.split(".", 2)[1]

    # Sometimes line number contains a redundant "l" at the end ("Q005624.1l" for example), so we ignore it.
    if n[-1] == "l":
        n = n[:-1]

    if not n.isdigit():
        return -1

    line_number = int(n)

    return line_number


def from_key_to_text_and_line_numbers(key):
    # Calculation of start_line and end_line for signs and transcriptions
    text = key[0].split(".", 2)[0]
    start_line = from_key_to_line_number(key[0])

    if start_line == -1:
        return text, -1, -1

    # Sometimes the end line is not specified when it's one line ("n057" for example), so we use the start line.
    if "." in key[1]:
        end_line = from_key_to_line_number(key[1])
    else:
        end_line = start_line

    return text, start_line, end_line


def parse_word(sentence, word):
    if word.text:
        sentence += word.text

    if len(word) >= 1:
        inner_sentence = ""
        for inner_word in word:
            inner_sentence = parse_word(inner_sentence, inner_word)

        sentence += inner_sentence

    if word.tail:
        sentence += word.tail

    return sentence


def handle_word_by_type(sentence, word):
    if word.attrib["type"] == "w" or word.attrib["type"] == "r" or word.attrib["type"] == "foreign" \
            or word.attrib["type"] == "i" or word.attrib["type"] == "smaller":
        sentence = parse_word(sentence, word)

    elif word.attrib["type"] == "bi" or word.attrib["type"] == "notelink":
        # "bi" seems like Akkadian prefixes, so we don't include it.
        # "notelink" seems like a number for footnotes, so no need to include it.
        pass

    else:
        print(word.attrib["type"])
        raise Exception("unknown type of word")

    return sentence


def collect_translations(div, translations):
    for tr in div:
        if "type" not in tr.attrib.keys():
            # [0] goes <span type="w">.
            language = tr[0].text
            # Seems like § was forgotten before 54, so added a special case for this (language[0].isdigit()).
            if language == "Akkadian" or language == "Date" or language == "Fragment" or language == "Colophon" \
                    or language == "Catch-line" or language[0] == "§" or language[0].isdigit() or \
                    language == "Envelope" or language == "Epigraph":
                tr = div[1]
            elif language == "Aramaic" or language == "Inscription":
                continue
            else:
                print(language)
                raise Exception("unknown language")

        if tr.attrib["type"] == "note":
            # "note" seems like a footnotes, so no need to include it.
            continue

        if tr.attrib["type"] != "tr":
            print(tr.attrib["type"])
            raise Exception("unknown type in div1")

        if tr.attrib["subtype"] == "tr":
            if "{http://oracc.org/ns/xtr/1.0}ref" in tr.attrib.keys():
                sref = tr.attrib["{http://oracc.org/ns/xtr/1.0}ref"]
                eref = tr.attrib["{http://oracc.org/ns/xtr/1.0}ref"]
            else:
                sref = tr.attrib["{http://oracc.org/ns/xtr/1.0}sref"]
                eref = tr.attrib["{http://oracc.org/ns/xtr/1.0}eref"]

            # [0][0] goes for <p> and <span type="cell">
            p = tr[0]
            if len(p) == 0:
                # No actual translation.
                continue

            sentence = ""
            for word in p[0]:
                sentence = handle_word_by_type(sentence, word)

            translations[(sref, eref)] = sentence

        elif tr.attrib["subtype"] == "dollar":
            # General comment in English, not translation
            continue

        else:
            print(tr.attrib["subtype"])
            raise Exception("unknown subtype of tr")

    return translations


def build_key(text, n):
    return text + "." + str(n)


def is_in_range(index, mapping, start, end):
    n = from_key_to_line_number(mapping[index])
    if start <= n and n <= end:
        return True
    return False


def index_in_mapping(index, mapping, start, end):
    if index in mapping:
        return True, index

    if index[-1] == "'" and index[:-1] in mapping:
        return True, index[:-1]

    if index[-1].isalpha() and index[:-1] in mapping:
        return True, index[:-1]

    if index.replace("rev.", "r") in mapping:
        return True, index.replace("rev.", "r")

    if index.replace("rev.", "r.") in mapping:
        return True, index.replace("rev.", "r.")

    if "o " + index in mapping:
        return True, "o " + index

    if "r " + index in mapping:
        return True, "r " + index

    if index.replace("vi'", "x") in mapping:
        return True, index.replace("vi'", "x")

    if "i " + index in mapping and is_in_range("i " + index, mapping, start, end):
        return True, "i " + index

    if "i " + index + "'" in mapping and is_in_range("i " + index + "'", mapping, start, end):
        return True, "i " + index + "'"

    if "ii " + index in mapping and is_in_range("ii " + index, mapping, start, end):
        return True, "ii " + index

    if "ii " + index + "'" in mapping and is_in_range("ii " + index + "'", mapping, start, end):
        return True, "ii " + index + "'"

    if "iii " + index in mapping and is_in_range("iii " + index, mapping, start, end):
        return True, "iii " + index

    if "iii " + index + "'" in mapping and is_in_range("iii " + index + "'", mapping, start, end):
        return True, "iii " + index + "'"

    if "iv " + index in mapping and is_in_range("iv " + index, mapping, start, end):
        return True, "iv " + index

    if "iv " + index + "'" in mapping and is_in_range("iv " + index + "'", mapping, start, end):
        return True, "iv " + index + "'"

    if "x " + index in mapping and is_in_range("x " + index, mapping, start, end):
        return True, "x " + index

    if "x " + index + "'" in mapping and is_in_range("x " + index + "'", mapping, start, end):
        return True, "x " + index + "'"

    return False, index


def divide_translation(raw_translations, line_mapping, corpus):
    translations = {}

    for key in raw_translations:
        t = raw_translations[key]

        # Canonize the apostrophe, erase square brackets and replace new lines with spaces.
        t = t.replace("´", "'").replace("′", "'").replace("[", "").replace("]", "").replace("\n", " ")

        # The translation doesn't contain any content, so we can't use it.
        if "No translation possible" in t or "No translation warranted" in t or "broken for translation" in t or \
                "fragmentary for translation" in t or t.replace(" ", "").replace(".", "") == "":
            continue

        text, start_line, end_line = from_key_to_text_and_line_numbers(key)

        if (corpus, text) not in line_mapping.keys():
            continue

        while "(" in t:
            split_par = t.split("(", 1)
            split_again = split_par[1].split(")", 1)
            index = split_again[0]

            is_in, index = index_in_mapping(index, line_mapping[(corpus, text)], start_line, end_line)

            if not is_in:
                t = split_par[0] + index + split_again[1]

            else:
                n = from_key_to_line_number(line_mapping[(corpus, text)][index])
                translations[(build_key(text, start_line), build_key(text, n-1))] = split_par[0]
                start_line = n
                t = split_again[1]

        translations[(build_key(text, start_line), build_key(text, end_line))] = t

    return translations


# This function returns translations of a full corpus
def parse_xml(file, line_mapping, corpus):
    raw_translations = {}
    tree = ET.parse(file)
    root = tree.getroot()

    for line in range(3, len(root)):
        #[1][0] goes for <text xml:lang="akk" ...> (after <teiHeader>) and <body>.
        body = root[line][1][0]
        if len(body) <= 1:
            # this means the text contains no translations at all.
            continue

        # [1] goes for <div1 xmlns:xtr="http://oracc.org/ns/xtr/1.0" ...> (after another <div1 type="discourse" ...>).
        div1 = body[1]
        if len(div1) == 0:
            # this means the text contains no translations at all.
            continue

        # Some translations are in German, so we take the English translation.
        if div1.attrib["{http://www.w3.org/XML/1998/namespace}lang"] == "de":
            div1 = body[2]

        if div1.attrib["{http://www.w3.org/XML/1998/namespace}lang"] != "en":
            print(div1.attrib["{http://www.w3.org/XML/1998/namespace}lang"])
            raise Exception("unknown language")

        # Sample one div.
        div = div1[0]
        if div.tag == "{http://www.tei-c.org/ns/1.0}div2":
            for div2 in div1:
                raw_translations = collect_translations(div2, raw_translations)
        elif div.tag == "{http://www.tei-c.org/ns/1.0}div3":
            raw_translations = collect_translations(div1, raw_translations)
        else:
            print(div)
            raise Exception("unknown div")

    translations = divide_translation(raw_translations, line_mapping, corpus)

    return translations