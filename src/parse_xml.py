import xml.etree.ElementTree as ET


def parse_word(sentence, word):
    sentence += word.text

    if len(word) >= 1:
        for suffix in word:
            if suffix.tail:
                sentence += suffix.tail

    if word.tail:
        sentence += word.tail

    return sentence


def handle_word_by_type(sentence, word):
    if word.attrib["type"] == "w" or word.attrib["type"] == "foreign":
        sentence = parse_word(sentence, word)

    elif word.attrib["type"] == "i" or word.attrib["type"] == "smaller":
        for inner_word in word:
            sentence = handle_word_by_type(sentence, inner_word)

    elif word.attrib["type"] == "r":
        # We don't want to include parenthesis in the translation ("(", ")", "[", "]") so we don't add text.
        if word.tail:
            sentence += word.tail
        pass

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
            # Seems like § was forgotten before 54, so added a special case for this.
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

            sentence = ""
            # [0][0] goes for <p> and <span type="cell">
            p = tr[0]
            if len(p) == 0:
                # No actual translation.
                continue
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


def parse_xml(file):
    translations = {}
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

        # Sample one div.
        div = div1[0]
        if div.tag == "{http://www.tei-c.org/ns/1.0}div2":
            for div2 in div1:
                translations = collect_translations(div2, translations)
        elif div.tag == "{http://www.tei-c.org/ns/1.0}div3":
            translations = collect_translations(div1, translations)
        else:
            print(div)
            raise Exception("unknown div")

    return translations