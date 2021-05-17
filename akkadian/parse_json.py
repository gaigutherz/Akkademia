import json
from pathlib import Path


def get_delim(c):
    """
    Gets the right delimiter from the c node
    :param c: the c node json object
    :return: delimiter of the transcription
    """
    if "delim" in c:
        return c["delim"]
    else:
        return None


def parse_tran(c, chars, key, add_three_dots):
    """
    Prase c node to fill chars up
    :param c: the c node json object
    :param chars: list collecting all characters (id, transliteration, delimiter, sign) from the .json file
    :param key: current key (text, start, end)
    :param add_three_dots: whether we should add three dots out of the .json file or not
    :return: nothing
    """
    if "group" in c:
        for elem in c["group"]:
            parse_tran(elem, chars, key, add_three_dots)

    elif "det" in c:
        if c["pos"] == "pre":
            delim = "-"
        else:
            delim = None
        c = c["seq"][0]
        if "group" in c:
            c = c["group"][0]
            if "s" in c:
                try:
                    chars.append([c["id"], "{" + c["s"] + "}", delim, c["utf8"]])
                except:
                    chars.append([c["id"], "{" + c["s"] + "}", delim, ""])
            # Very rare! Reaches this case only in saao.
            else:
                try:
                    chars.append([c["id"], "{" + c["form"] + "}", delim, c["utf8"]])
                except:
                    chars.append([c["id"], "{" + c["form"] + "}", delim, ""])

        else:
            if "v" in c:
                try:
                    chars.append([c["id"], "{" + c["v"] + "}", delim, c["utf8"]])
                except:
                    chars.append([c["id"], "{" + c["v"] + "}", delim, ""])
            elif "s" in c:
                try:
                    chars.append([c["id"], "{" + c["s"] + "}", delim, c["utf8"]])
                except:
                    chars.append([c["id"], "{" + c["s"] + "}", delim, ""])
            # Very rare! Reaches this case only in saao.
            elif "sexified" or "form" in c:
                try:
                    chars.append([c["id"], "{" + c["form"] + "}", delim, c["utf8"]])
                except:
                    chars.append([c["id"], "{" + c["form"] + "}", delim, ""])
            else:
                print(c)
                raise Exception("c doesn't contain v / s / sexified / form! We don't know how to parse it!")

    elif "sexified" in c:
        try:
            chars.append([c["id"], c["form"], get_delim(c), c["utf8"]])
        except:
            chars.append([c["id"], c["form"], get_delim(c), ""])

    elif "v" in c:
        # There is a mistake in X900016.json in saao02 where there is no id, so ignoring it.
        if "id" in c:
            try:
                chars.append([c["id"], c["v"], get_delim(c), c["utf8"]])
            except:
                chars.append([c["id"], c["v"], get_delim(c), ""])

    elif "s" in c:
        # There is a mistake in P336656.json in saao08 where there is no id, so ignoring it.
        if "id" in c:
            try:
                chars.append([c["id"], c["s"], get_delim(c), c["utf8"]])
            except:
                chars.append([c["id"], c["s"], get_delim(c), ""])

    # Very rare!
    elif "q" in c:
        try:
            chars.append([c["id"], c["q"], get_delim(c), c["utf8"]])
        except:
            try:
                chars.append([c["id"], c["q"], get_delim(c), c["qualified"][1]["utf8"]])
            except:
                chars.append([c["id"], c["q"], get_delim(c), ""])

    # Very rare!
    elif "c" in c:
        try:
            chars.append([c["id"], c["c"], get_delim(c), c["utf8"]])
        except:
            chars.append([c["id"], c["c"], get_delim(c), ""])

    # Very rare! Doesn't appear in rinap, only riao (the letter "n").
    elif "n" in c:
        try:
            chars.append([c["id"], c["n"], get_delim(c), c["utf8"]])
        except:
            chars.append([c["id"], c["n"], get_delim(c), ""])

    # Very rare! Appears only in saao (the letter "form" when there is no other letter).
    elif "form" in c:
        try:
            chars.append([c["id"], c["form"], get_delim(c), c["utf8"]])
        except:
            chars.append([c["id"], c["form"], get_delim(c), ""])

    # Very rare! Appears only in saao (the letter "form" when there is no other letter).
    elif "p" in c:
        try:
            chars.append([c["id"], c["p"], get_delim(c), c["utf8"]])
        except:
            chars.append([c["id"], c["p"], get_delim(c), ""])

    # Broken sign, doesn't interest us
    elif "x" in c:
        if add_three_dots and c["x"] == "ellipsis":
            chars.append([c["id"], "...", None, "..."])

    # Happened in P429046.json
    elif "gloss" in c:
        pass

    else:
        print(c)
        raise Exception("c doesn't contain group / det / sexified / v / s / q / c / n / form / p / x! We don't know how to parse it!")


def parse_translation(l_node, translation):
    """
    Parse an l node inside the .json file for getting a sign direct translation
    :param l_node: the l node json object
    :param translation: list collecting all translations of signs from the .json file
    :return: nothing
    """
    if "sense" not in l_node["f"]:
        return

    t = l_node["f"]["sense"]
    if t == "1":
        if "norm" in l_node["f"]:
            t = l_node["f"]["norm"]
        else:
            t = l_node["f"]["norm0"]

    translation.append([l_node["ref"], t])


def parse_l_node(l_node, chars, translation, key, add_three_dots):
    """
    Parse an l node inside the .json file
    :param l_node: the l node json object
    :param chars: list collecting all characters (id, transliteration, delimiter, sign) from the .json file
    :param translation: list collecting all translations of signs from the .json file
    :param key: current key (text, start, end)
    :param add_three_dots: whether we should add three dots out of the .json file or not
    :return: nothing
    """
    if l_node["f"]["lang"] == "arc" or l_node["f"]["lang"] == "qcu-949" or l_node["f"]["lang"] == "akk-949" \
            or l_node["f"]["lang"] == "arc-949" or l_node["f"]["lang"] == "akk-x-neoass-949":
        return

    parse_translation(l_node, translation)
    if "gdl" not in l_node["f"]:
        print(l_node)
        exit()

    gdl = l_node["f"]["gdl"]
    for g in gdl:
        parse_tran(g, chars, key, add_three_dots)


def parse_d_node(node, mapping):
    """
    Parse a d node inside the .json file
    :param node: the d node json object
    :param mapping: dictionary collecting mapping from two types of line numbering
    :return: nothing
    """
    # If we're not at a line start, we don't have the mapping we are interested in.
    if node["type"] != "line-start":
        return

    # Probably a duplicate d node, and next one will contain the mapping.
    if "label" not in node:
        return

    if "ref" not in node:
        print(node)
        raise Exception("We reached a line-start node with no ref")

    # Sometimes ref appear with a redundant l at the end, such as "Q005624.9l".
    ref = node["ref"]
    if ref[-1] == "l":
        ref = ref[:-1]

    mapping[node["label"]] = ref


def parse_c_node(c_node, chars, translation, mapping, key, lines_cut_by_translation, add_three_dots):
    """
    Parse a c node inside the .json file
    :param c_node: the c node json object
    :param chars: list collecting all characters (id, transliteration, delimiter, sign) from the .json file
    :param translation: list collecting all translations of signs from the .json file
    :param mapping: dictionary collecting mapping from two types of line numbering
    :param key: current key (text, start, end)
    :param lines_cut_by_translation: list of lines cut by translation
    :param add_three_dots: whether we should add three dots out of the .json file or not
    :return: nothing
    """
    # This text doesn't have data in it.
    if "cdl" not in c_node:
        return

    if c_node["type"] == "sentence":
        if c_node["cdl"][0]["node"] != "d":
            if c_node["cdl"][0]["node"] == "c":
                lines_cut_by_translation.append(c_node["cdl"][0]["cdl"][0]["ref"])
            else:
                lines_cut_by_translation.append(c_node["cdl"][0]["ref"])

        text = c_node["id"].split(".")[0]
        if "label" not in c_node:
            start = "all"
            end = "all"

        elif " - " in c_node["label"]:
            labels = c_node["label"].split(" - ")
            start = labels[0]
            end = labels[1]

        else:
            start = c_node["label"]
            end = start

        key = (text, start, end)

    for node in c_node["cdl"]:
        # This means it is metadata, and not a fragment.
        if node["node"] == "d":
            parse_d_node(node, mapping)

        elif node["node"] == "c":
            parse_c_node(node, chars, translation, mapping, key, lines_cut_by_translation, add_three_dots)

        elif node["node"] == "l":
            parse_l_node(node, chars, translation, key, add_three_dots)

        # Very rare! Appears only in saao (choice between two options).
        elif node["node"] == "ll":
            parse_l_node(node["choices"][0], chars, translation, key, add_three_dots)

        else:
            print(node)
            raise Exception("We reached a node other than d / c / l / ll. We don't know how to parse it!")


def process_cut_lines(lines_cut_by_translation):
    """
    Collect all of the special lines we have (lines that were cut in the middle during translation)
    :param lines_cut_by_translation: list of lines cut by translation
    :return: list of special lines
    """
    special_lines = []

    for line in lines_cut_by_translation:
        values = line.split(".")
        cut_line = values[0] + "." + values[1]
        threshold = int(values[2])
        special_lines.append([cut_line, threshold])

    return special_lines


def parse_json(file, add_three_dots=False):
    """
    Parses a .json file to get the data we need out of it
    :param file: .json file for parsing
    :param add_three_dots: whether we should add three dots out of the .json file or not
    :return: all values we are interested in from the .json file
    """
    chars = []
    translation = []
    mapping = {}
    lines_cut_by_translation = []

    f = open(file, "r", encoding="utf8")
    data = f.read()

    if not data:
        return None, None, None, None

    json_object = json.loads(data)
    j = json_object["cdl"][0]
    parse_c_node(j, chars, translation, mapping, None, lines_cut_by_translation, add_three_dots)

    if chars == [] and translation == [] and mapping == {}:
        return None, None, None, None

    lines_cut_by_translation = process_cut_lines(lines_cut_by_translation)
    return chars, translation, mapping, lines_cut_by_translation


def main():
    """
    Intended to try the logic of the file for tests.
    :return: nothing
    """
    directory = Path(r"../raw_data/rinap/rinap1/")
    chars, translation, mapping, line_cut_by_translation = parse_json(directory / "Q003431.json", True)

    print(len(chars))
    for c in chars:
        print(c[1])

    print(len(translation))
    for t in translation:
        print(t[1])

    print(mapping)
    print(line_cut_by_translation)


if __name__ == '__main__':
    main()
