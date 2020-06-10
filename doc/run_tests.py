#!/usr/bin/env python3

import os
import collections

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
            cvs.writePDFfile("images/%s.pdf" % name)
            cvs.writeSVGfile("images/%s.svg" % name)
            print()
        except:
            print("render failed for %r"%name)
            raise

      except StopIteration:
        break

    return lnos


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


def main():

    path = os.path.dirname(__file__)
    names = os.listdir(path)
    names = [name for name in names 
        if name.endswith(".py") and name.startswith("test_")]
    names.sort()

    for name in names:
        process(path, name)


if __name__ == "__main__":
    main()


