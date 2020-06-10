#!/usr/bin/env python3

import os
import collections

from pygments.formatters import HtmlFormatter
from pygments import highlight
from pygments.lexers import Python3Lexer

from bruhat.argv import argv
from bruhat.render.doc import run_tests
from bruhat.render.front import Canvas, Scale


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


# This is like recursive descent parsing:

def process(path, name):

    fullname = os.path.join(path, name)
    
    output = open("output.html", 'w')
    style = HtmlFormatter().get_style_defs('.highlight') 
    print(html_head(html_style(style)), file=output)

    code = open(fullname).read().split('\n')

    for test in run_tests.harvest(path, name):
        snip = code[test.start : test.end]

        for block in html_snip(snip):
            print(block, file=output)

        print(html_img(test.name), file=output)

    print(html_tail(), file=output)
    output.close()


def html_code(lines):
    s = '\n'.join(lines)
    if not s.strip():
        return ""
    return highlight(s, Python3Lexer(), HtmlFormatter())

def html_comment(lines):
    lines = [line[1:] for line in lines]
    s = '\n'.join(lines)
    return html_p(s)


def html_snip(lines):

    lines = dedent(lines)

    code = []
    comment = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("#"):
            if code:
                yield html_code(code)
                code = []
            comment.append(line)
        else:
            if comment:
                yield html_comment(comment)
                comment = []
            code.append(line)
        idx += 1

    if code:
        yield html_code(code)

    if comment:
        yield html_comment(comment)



def dedent(lines):
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
    return lines


def main():
    path = "."
    names = "test_boxs.py".split()
    for name in names:
        process(path, name)


if __name__ == "__main__":
    main()


