import json
import platform

def main():
    if platform.system() == "Windows":
        directory = r"raw_data\rinap\rinap1\corpusjson"
        chars = parse_json("..\\" + directory + "\\" + "Q003453.json")
    else:
        directory = r"raw_data/rinap/rinap1/corpusjson"
        chars = parse_json("../" + directory + "/" + "Q003453.json")
    print(len(chars))
    for c in chars:
        print(c[1])


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
            chars.append([c["id"], "{" + c["s"] + "}", delim, c["utf8"]])
        else:
            if "v" in c:
                chars.append([c["id"], "{" + c["v"] + "}", delim, c["utf8"]])
            elif "s" in c:
                chars.append([c["id"], "{" + c["s"] + "}", delim, c["utf8"]])
            else:
                raise Exception("c doesn't contain v / s! We don't know how to parse it!")

    elif "sexified" in c:
        chars.append([c["id"], c["form"], get_delim(c), c["utf8"]])

    elif "v" in c:
            chars.append([c["id"], c["v"], get_delim(c), c["utf8"]])

    elif "s" in c:
            chars.append([c["id"], c["s"], get_delim(c), c["utf8"]])

    # Very rare!
    elif "q" in c:
            chars.append([c["id"], c["q"], get_delim(c), c["utf8"]])

    # Very rare!
    elif "c" in c:
            chars.append([c["id"], c["c"], get_delim(c), c["utf8"]])

    # Broken sign, doesn't interest us
    elif "x" in c:
        pass

    else:
        raise Exception("c doesn't contain group / det / sexified / v / s / q / x! We don't know how to parse it!")


def parse_l_node(l_node, chars):
    if l_node["f"]["lang"] == "arc" or l_node["f"]["lang"] == "qcu-949" or l_node["f"]["lang"] == "akk-949":
        return
    gdl = l_node["f"]["gdl"]
    for g in gdl:
        parse_tran(g, chars)


def parse_c_node(c_node, chars):
    for node in c_node["cdl"]:
        # This means it is metadata, and not a fragment.
        if node["node"] == "d":
            continue
        elif node["node"] == "c":
            parse_c_node(node, chars)
        elif node["node"] == "l":
            parse_l_node(node, chars)
        else:
            raise Exception("We reached a node other than d / c / l. We don't know how to parse it!")


def parse_json(file):
    chars = []
    f = open(file, "r", encoding="utf8")
    data = f.read()

    json_object = json.loads(data)
    j = json_object["cdl"][0]
    parse_c_node(j, chars)
    return chars


if __name__ == '__main__':
    main()
