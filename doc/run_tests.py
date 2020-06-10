#!/usr/bin/env python3

import os
import collections

import bruhat.render.doc
from bruhat.render.front import Canvas, Scale


class TestRun(object):
    def __init__(self, func, start, end, name):
        self.func = func
        self.start = start
        self.end = end
        self.name = name




def run_test(func, dummy=False):

    items = func()

    if items is None:
        return

    start = items.gi_frame.f_lineno # index

    while 1:
      try:
        box, name = items.__next__()
        end = items.gi_frame.f_lineno-1 # index

        svgname = "images/%s.svg"%name
        pdfname = "images/%s.pdf"%name
        if not dummy:
            try:
                print("run_tests: rendering", name)
                cvs = Canvas()
                cvs.append(Scale(2.0))
                box.render(cvs)
                cvs.writeSVGfile(svgname)
                cvs.writePDFfile(pdfname)
                print()
            except:
                print("run_tests: render failed for %r"%name)
                raise

        test = TestRun(func, start, end, svgname)
        yield test

        start = end+1

      except StopIteration:
        break



def harvest(path, name, dummy=False):
    print("run_tests.harvest", name)
    assert name.endswith(".py")
    stem = name[:-len(".py")]
    desc = "bruhat.render.doc."+stem
    __import__(desc)
    m = getattr(bruhat.render.doc, stem)
    funcs = []
    for attr in dir(m):
        value = getattr(m, attr)
        if attr.startswith("test_") and isinstance(value, collections.Callable):
            funcs.append(value)

    funcs.sort(key = lambda f : (f.__module__, f.__code__.co_firstlineno))
    for func in funcs:
        for test in run_test(func, dummy=dummy):
            yield test


def run():

    path = os.path.dirname(__file__)
    names = os.listdir(path)
    names = [name for name in names 
        if name.endswith(".py") and name.startswith("test_")]
    names.sort()

    for name in names:
        for test in harvest(path, name):
            yield test


def main():
    for test in run():
        pass

    print("run_tests.main: finished")


if __name__ == "__main__":
    main()


