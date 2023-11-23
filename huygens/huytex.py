#!/usr/bin/env python3

"""
replace %py lines with appropriate latex commands,
and exec the %py 
"""

import os
import hashlib

def exists(name):
    try:
        open(name).close()
        return True
    except OSError:
        return False


#import config
#config.weld = False
#from config import *


state = []
stems = set()
lookup = {}

match_py = "%py "
match_PY = "%PY "
GRI = r"\raisebox{-0.4\height}{\includegraphics{%s/%s.pdf}}"
GRD = r"{\begin{center}\includegraphics{%s/%s.pdf}\end{center}}"

suspend = False
def process(line, ns):
    global suspend
    if "get_latex" not in ns:
        ns["get_latex"] = get_latex

    if match_py in line and line.index("%") == line.index(match_py):
        idx = line.index(match_py)
        cmd = GRI
        #size = 0.4
    elif match_PY in line and line.index("%") == line.index(match_PY):
        idx = line.index(match_PY)
        cmd = GRD
        #size = 1.0
    else:
        return line
    
    pre = line[:idx]
    post = line[idx+4:]

    if not post.strip():
        # nothing to do here
        return pre+"\n" if pre else pre # <------ return

    if post.strip() == "%py exit":
        suspend = True # arfff...
        return pre+"\n" if pre else pre # <------ return

    data = state
    try:
        #print("eval: %r"%post)
        value = eval(post, ns, ns)
        if value is None: # function call...
            state.append(post)
            return pre+"\n" if pre else pre # <------ return
        # WARNING: we _assume expr's don't mutate state
        # otherwise, uncomment the next line:
        #state.append(post) 
        data = state + [post]
    except SyntaxError:
        exec(post, ns, ns)
        state.append(post)
        value = None
        return pre+"\n" if pre else pre # <------ return
    except:
        print("proc.py failed at %r"%(post,))
        raise

    if hasattr(value, "_latex_"):
        value = value._latex_()

    if type(value) is str:
        return pre + value
        #value = value.replace("_", r"\_")
        #return r"{\tt %s}"%(value,)

    data = '\n'.join(data)
    data = data.encode('utf-8')
    stem = hashlib.sha1(data).hexdigest()

    if stem not in stems:
        name = "%s/%s.pdf"%(imdir, stem,)
        if not exists(name):
            try:
                save(stem, value, imdir=imdir) # monkeypatch from local config
            except:
                print("proc.py failed at %r"%(post,))
                raise
        stems.add(stem)

    latex = cmd%(imdir, stem,) 
    lookup[id(value)] = latex
    line = pre + latex + "\n"

    return line


def get_latex(value, cmd=GRI, **kw):
    key = id(value)
    if key in lookup:
        return lookup[key]
    data = str(key).encode('utf-8') # ?
    stem = hashlib.sha1(data).hexdigest()
    save(stem, value, imdir=imdir, **kw) # monkeypatch from local config
    latex = cmd%(imdir, stem,) 
    lookup[id(value)] = latex
    return latex

    
def main(tgt, src, ns={}):
    global imdir

    f = open(src)
    if exists(tgt):
        print("file %r exists!"%tgt)
        return

    g = open(tgt, 'w')
    
    assert tgt.endswith(".tex")
    imdir = tgt[:-len(".tex")] + ".imcache"
    try:
        os.mkdir(imdir)
    except FileExistsError:
        pass

    for line in f.readlines():
        line = process(line, ns)
        print(line, end="", file=g)

        if line.strip() == r"\end{document}":
            break


if __name__ == "__main__":

    import sys
    for item in sys.argv[3:]:
        state.append(open(item).read())

    main(sys.argv[1], sys.argv[2])

