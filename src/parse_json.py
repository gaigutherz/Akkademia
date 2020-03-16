import json
from pathlib import Path


def get_delim(c):
    if "delim" in c:
        return c["delim"]
    else:
        return None


def parse_tran(c, chars):
    if "group" in c:
        for elem in c["group"]:
            parse_tran(elem, chars)

    elif "det" in c:
        if c["pos"] == "pre":
            delim = "-"
        else:
            delim = None
        c = c["seq"][0]
        if "group" in c:
            c = c["group"][0]
            if "s" in c:
                chars.append([c["id"], "{" + c["s"] + "}", delim, c["utf8"]])
            # Very rare! Reaches this case only in saao.
            else:
                chars.append([c["id"], "{" + c["form"] + "}", delim, c["utf8"]])

        else:
            if "v" in c:
                chars.append([c["id"], "{" + c["v"] + "}", delim, c["utf8"]])
            elif "s" in c:
                chars.append([c["id"], "{" + c["s"] + "}", delim, c["utf8"]])
            # Very rare! Reaches this case only in saao.
            elif "sexified" or "form" in c:
                chars.append([c["id"], "{" + c["form"] + "}", delim, c["utf8"]])
            else:
                print(c)
                raise Exception("c doesn't contain v / s / sexified / form! We don't know how to parse it!")

    elif "sexified" in c:
        chars.append([c["id"], c["form"], get_delim(c), c["utf8"]])

    elif "v" in c:
        # There is a mistake in X900016.json in saao02 where there is no id, so ignoring it.
        if "id" in c:
            chars.append([c["id"], c["v"], get_delim(c), c["utf8"]])

    elif "s" in c:
        # There is a mistake in P336656.json in saao08 where there is no id, so ignoring it.
        if "id" in c:
            chars.append([c["id"], c["s"], get_delim(c), c["utf8"]])

    # Very rare!
    elif "q" in c:
        try:
            chars.append([c["id"], c["q"], get_delim(c), c["utf8"]])
        except:
            chars.append([c["id"], c["q"], get_delim(c), c["qualified"][1]["utf8"]])

    # Very rare!
    elif "c" in c:
        chars.append([c["id"], c["c"], get_delim(c), c["utf8"]])

    # Very rare! Doesn't appear in rinap, only riao (the letter "n").
    elif "n" in c:
        chars.append([c["id"], c["n"], get_delim(c), c["utf8"]])

    # Very rare! Appears only in saao (the letter "form" when there is no other letter).
    elif "form" in c:
        chars.append([c["id"], c["form"], get_delim(c), c["utf8"]])

    # Very rare! Appears only in saao (the letter "form" when there is no other letter).
    elif "p" in c:
        chars.append([c["id"], c["p"], get_delim(c), c["utf8"]])

    # Broken sign, doesn't interest us
    elif "x" in c:
        pass

    else:
        print(c)
        raise Exception("c doesn't contain group / det / sexified / v / s / q / c / n / form / p / x! We don't know how to parse it!")


def parse_translation(l_node, translation):
    if "sense" not in l_node["f"]:
        return

    t = l_node["f"]["sense"]
    if t == "1":
        if "norm" in l_node["f"]:
            t = l_node["f"]["norm"]
        else:
            t = l_node["f"]["norm0"]

    translation.append([l_node["ref"], t])


def parse_l_node(l_node, chars, translation):
    if l_node["f"]["lang"] == "arc" or l_node["f"]["lang"] == "qcu-949" or l_node["f"]["lang"] == "akk-949" \
            or l_node["f"]["lang"] == "arc-949":
        return

    parse_translation(l_node, translation)
    if "gdl" not in l_node["f"]:
        print(l_node)
        exit()
    gdl = l_node["f"]["gdl"]
    for g in gdl:
        parse_tran(g, chars)


def parse_c_node(c_node, chars, translation):
    for node in c_node["cdl"]:
        # This means it is metadata, and not a fragment.
        if node["node"] == "d":
            continue
        elif node["node"] == "c":
            parse_c_node(node, chars, translation)
        elif node["node"] == "l":
            parse_l_node(node, chars, translation)
        # Very rare! Appears only in saao (choice between two options).
        elif node["node"] == "ll":
            parse_l_node(node["choices"][0], chars, translation)
        else:
            print(node)
            raise Exception("We reached a node other than d / c / l / ll. We don't know how to parse it!")


def parse_json(file):
    chars = []
    translation = []

    f = open(file, "r", encoding="utf8")
    data = f.read()

    if not data:
        return None, None

    json_object = json.loads(data)
    j = json_object["cdl"][0]
    parse_c_node(j, chars, translation)
    return chars, translation


def main():
    directory = Path(r"../raw_data/rinap/rinap4/")
    chars, translation = parse_json(directory / "Q003355.json")

    print(len(chars))
    for c in chars:
        print(c[1])

    print(len(translation))
    for t in translation:
        print(t[1])


if __name__ == '__main__':
    main()
