#!/usr/bin/env python3

import os
import hashlib

from bruhat.render import load_svg

def file_exists(name):
    try:
        os.stat(name)
        return True
    except:
        pass
    return False


def command(cmd):
    ret = os.system(cmd)
    assert ret == 0, "%r failed with return value %d"%(cmd, ret)


def make_text(text):
    cache = "__bruhat__"
    if not file_exists(cache):
        os.mkdir(cache)
    os.chdir(cache) # <---------- chdir <-----

    data = text.encode('utf-8')
    stem = hashlib.sha1(data).hexdigest()

    tex = "%s.tex"%stem
    svg = "%s.svg"%stem
    pdf = "%s.pdf"%stem

    if not file_exists(svg):

        f = open(tex, 'w')
    
        # if you change this don't forget to delete the cache 
        print(r"\def\folio{}", file=f)
        print(r"%s\bye"%text, file=f)
        f.close()
    
        command("pdftex %s"%tex)
        command("pdf2svg %s %s" % (pdf, svg))

    item = load_svg.load(svg)

    os.chdir("..") # <---------- chdir <-----

    return item


def test():

    item = make_text("hi there!")
    print(item)


if __name__ == "__main__":

    test()

