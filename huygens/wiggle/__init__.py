
from huygens import config, back
if back.the_text_cls is back.CairoText:
    config(text="pdflatex", latex_header=r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    """)

import huygens
if huygens.EXPERIMENTAL:
    from huygens.wiggle.cell_experimental import *
else:
    from huygens.wiggle.cell import *

from huygens.wiggle.shapes import *

