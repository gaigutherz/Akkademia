from pathlib import Path
import os
import requests
import xml.etree.ElementTree as ET


def get_dialect(corpus, file):
    d = requests.get('http://oracc.museum.upenn.edu/' + corpus + '/' + file[:-5] + '/')
    try:
        root = ET.fromstring(d.content)
    except:
        return None

    i = 0
    for child in root.iter('{http://www.w3.org/1999/xhtml}li'):
        i += 1
        if i == 9:
            s = child.text
        if i == 10:
            if child.text[:7] == "Written":
                return s
            else:
                return child.text


def main():
    directory = Path(r"../raw_data/test_texts")
    corpus = "riao"

    for file in os.listdir(directory / corpus):
            print(file)
            print(get_dialect(corpus, file))


if __name__ == '__main__':
    main()
