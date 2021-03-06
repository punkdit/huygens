from huygens import front, back

from huygens.front import canvas, style, path, color, trafo, linestyle
from huygens import box, diagram
import huygens.text

tex_header = None
latex_header = None
xelatex_header = None


def config(text=None, tex_header=None, latex_header=None, xelatex_header=None):

    if text is None:
        pass
    elif text == "cairo":
        back.the_text_cls = back.CairoText
    elif text in "pdftex xetex xelatex pdflatex".split():
        back.the_text_cls = back.MkText
        back.MkText.tex_engine = text
    else:
        raise Exception("config text option %r not understood" % text)

    huygens.text.tex_header = tex_header
    huygens.text.latex_header = latex_header
    huygens.text.xelatex_header = xelatex_header


