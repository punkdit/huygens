#!/usr/bin/env python3

import os
import collections

import markdown
from pygments.formatters import HtmlFormatter
from pygments import highlight
from pygments.lexers import Python3Lexer

from huygens.argv import argv
from huygens.doc import run_tests
from huygens.box import Box


def html_head(s):
    html = """\
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" type="text/css" href="superstyle.css"> 
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Spectral">
%s
<style>
 p {
     font-family: 'Spectral', serif;
     font-size: 16px;
 }

 pre { font-size: 14px; }

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

def html_pre(body):
    return "<pre>%s</pre>\n"%body

def html_img(name):
    return '<p><img src="%s" class="center"></p>\n'%name


# This is like recursive descent parsing:

def main():
    dummy = argv.dummy
    select = argv.get("select", "")
    path = "."
    names = "test_canvas.py test_turtle.py test_sat.py test_box.py test_diagram.py"
    names = names.split()
    for name in names:
        if select in name:
            process(path, name, dummy)


def process(path, name, dummy=False):

    print("mkdoc.proces(%r, %r)"%(path, name))
    fullname = os.path.join(path, name)
    code = open(fullname).read().split('\n')

    assert fullname.endswith(".py")
    output = fullname[:-len(".py")] + ".html"
    
    output = open(output, 'w')
    style = HtmlFormatter().get_style_defs('.highlight') 
    print(html_head(html_style(style)), file=output)

    func = None
    test = None
    start = None

    try:
      for test in run_tests.harvest(path, name, dummy=dummy):

        if func is None:
            # first func in test script
            func = test.func

        if test.func is not func:
            end = find_dedent(code, start)
            #print(html_p(str((start, end))), file=output) # DEBUG
            snip = code[start+1: end]
            for block in html_snip(snip):
                print(block, file=output)
            print("\n\n<hr />\n", file=output)
            func = test.func

        #print(html_pre(test), file=output)

        end = test.end or find_dedent(code, test.start)

        # DEBUG
        #print(html_p("start=%s, end=%s" % (test.start, end)), file=output)
        snip = code[test.start : end]
        for block in html_snip(snip):
            print(block, file=output)

        if test.img:
            print(html_img(test.img), file=output)

        start = end
      if test and test.end:
        start = end
        end = find_dedent(code, start)
        if end > start:
            snip = code[start+1 : end]
            for block in html_snip(snip):
                print(block, file=output)
      #print(html_p("end of func"), file=output)

    except Exception:
        import traceback
        print("\n<pre>", file=output)
        traceback.print_exc(file=output)
        print("</pre>", file=output)
        traceback.print_exc()

    print(html_tail(), file=output)
    output.close()
    Box.DEBUG = False


def html_code(lines):
    s = '\n'.join(lines)
    if not s.strip():
        return ""
    return highlight(s, Python3Lexer(), HtmlFormatter())

COMMENT = "# "

def html_comment(lines):
    lines = [line[len(COMMENT):] for line in lines]
    s = '\n'.join(lines)
    #return html_p(s)

    md = markdown.Markdown()
    html = md.convert(s)
    return html


def html_snip(lines):

    lines = dedent(lines)

    code = []
    comment = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith(COMMENT):
            if code:
                yield html_code(code)
                code = []
            comment.append(line)
        elif line.startswith("#") and comment:
            assert not code
            comment.append(line)
        elif line.startswith("#"):
            pass
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


def get_indent(line):
    i = 0
    while line.startswith(' '*i):
        i += 1
    space = ' '*(i-1)
    return space


def find_dedent(lines, idx):

    space = None
    while idx < len(lines):
        line = lines[idx]
        if line.strip():
            indent = get_indent(lines[idx])
            if space is None:
                space = indent
            if len(indent) < len(space):
                break
        idx += 1
    return idx


def dedent(lines):
    indent = None
    assert lines
    for line in lines:
        if not line.strip():
            continue
        space = get_indent(line)
        if indent is None or indent.startswith(space):
            indent = space
    if indent is None:
        return []
    lines = [line[len(indent):] for line in lines]
    return lines


if __name__ == "__main__":
    main()
    print("mkdoc: OK")


