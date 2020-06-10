#!/usr/bin/env python3

import os
import collections

from bruhat.argv import argv
import bruhat.render.doc
from bruhat.render.front import Canvas, Scale


def run_test(func):

    i = func()
    lnos = [(i.gi_frame.f_lineno,None)]

    #for (box, name) in func():

    while 1:
      try:
        box, name = i.__next__()
        lnos.append((i.gi_frame.f_lineno, name))
        try:
            print("rendering", name)
            cvs = Canvas()
            cvs.append(Scale(2.0))
            box.render(cvs)
            #cvs = Canvas([Scale(2.0, 2.0)]+cvs.items)
            cvs.writePDFfile("pic-%s.pdf" % name)
            cvs.writeSVGfile("pic-%s.svg" % name)
            print()
        except:
            print("render failed for %r"%name)
            raise

      except StopIteration:
        break

    return lnos

def html_head(s):
    html = """\
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" type="text/css" href="superstyle.css"> 
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Spectral">
<link rel="stylesheet" type="text/css" href="superstyle.css"> 
%s
<style>
 body {
     font-family: 'Spectral', serif;
     font-size: 16px;
 }

 .center {
     display: block;
     margin-left: auto;
     margin-right: auto;
 }
</style>
</head>
<body>
<main>
""" % s
    return html

def html_tail():
    return """
<hr>

Copyright (c) 2018 - 2020.

</main>

</body>
"""

def html_style(body):
    return "<style>\n%s\n</style>\n"%body

def html_p(body):
    return "<p>%s</p>\n"%body

def html_img(name):
    return '<p><img src="%s" class="center"></p>\n'%name



def process(path, name):
    print("process", name)
    fullname = os.path.join(path, name)
    stem = name[:-len(".py")]
    desc = "bruhat.render.doc."+stem
    __import__(desc)
    m = getattr(bruhat.render.doc, stem)
    #print(desc, m)
    funcs = []
    for attr in dir(m):
        value = getattr(m, attr)
        if attr.startswith("test_") and isinstance(value, collections.Callable):
            funcs.append(value)

    funcs.sort(key = lambda f : (f.__module__, f.__code__.co_firstlineno))
    for func in funcs:
        lnos = run_test(func)

    if not argv.makedoc:
        return

    from pygments.formatters import HtmlFormatter
    
    output = open("output.html", 'w')
    style = HtmlFormatter().get_style_defs('.highlight') 
    print(html_head(html_style(style)), file=output)


    code = open(fullname).read().split('\n')
    lnos = [(lno-1,name) for (lno,name) in lnos]
    idx = 0
    while idx+1 < len(lnos):
        snip = code[lnos[idx][0]+1 : lnos[idx+1][0]]
        highlight(snip, output)
        print(html_img("pic-%s.svg"%lnos[idx+1][1]), file=output) # XXX D.R.Y.
        idx += 1

    print(html_tail(), file=output)
    output.close()


def highlight(lines, output):
    indent = None
    assert lines
    for line in lines:
        if not line.strip():
            continue
        i = 0
        while line.startswith(' '*i):
            i += 1
        space = ' '*(i-1)
        if indent is None or indent.startswith(space):
            indent = space
    assert indent is not None
    lines = [line[len(indent):] for line in lines]

    code = '\n'.join(lines)
    #print(code)

    from pygments import highlight
    from pygments.lexers import Python3Lexer
    from pygments.formatters import HtmlFormatter
    
    #code = 'print "Hello World"'
    print(highlight(code, Python3Lexer(), HtmlFormatter()), file=output)
    print(file=output)



def main():

    path = os.path.dirname(__file__)
    names = os.listdir(path)
    names = [name for name in names 
        if name.endswith(".py") and name.startswith("test_")]

    for name in names:
        process(path, name)


if __name__ == "__main__":
    main()


