from huygens import front, back

from huygens.front import canvas, style, path, color, trafo, linestyle
from huygens import box, diagram


def config(text=None):

    if text is None:
        pass
    elif text == "cairo":
        back.the_text_cls = back.CairoText
    elif text in "pdftex xetex xelatex pdflatex".split():
        back.the_text_cls = back.MkText
        back.MkText.tex_engine = text
    else:
        print("config text option %r not understood" % text)
        raise Exception



