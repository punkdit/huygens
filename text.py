#!/usr/bin/env python3

import os
import hashlib

from bruhat.render import load_svg

def make_cachedir(name="__bruhat__"):
    try:
        os.stat(name)
        return
    except:
        pass
    os.mkdir(name)


def make_text(text):
    cache = "__bruhat__"
    make_cachedir(cache)
    os.chdir(cache)

    data = text.encode('utf-8')
    stem = hashlib.sha1(data).hexdigest()

    f = open("%s.tex"%stem, 'w')
    print(r"\def\folio{}", file=f)
    print(r"%s\bye"%text, file=f)
    f.close()

    ret = os.system("pdftex %s.tex"%stem)
    assert ret == 0, ret

    ret = os.system("pdf2svg %s.pdf %s.svg"%(stem, stem))

    item = load_svg.load("%s.svg"%stem)

    os.chdir("..")

    return item


def test():

    item = make_text("hi there!")
    print(item)


if __name__ == "__main__":

    test()

