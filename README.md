
<img src="huygens/doc/images/Christiaan_Huygens.png" align="right" />

huygens
=======

This is a python package for drawing diagrams.
Intended to have multiple backends, currently the only
backend implemented uses [cairo](https://www.cairographics.org/).
Hopefully there will also be a [PyX](https://pyx-project.org/) backend.

It can also include text from $\TeX$ via pdftex, pdflatex, xetex or xelatex.

Diagrams are able to be composed in various ways, based 
on a linear constraint solver. Therefore, this system becomes
a declarative graphics layout package.

This package is at an early stage of development, so expect
things to change drastically and without warning.

User guide
----------

The documentation is in [huygens/doc](huygens/doc/). 
You can also read it online 
[here](https://arrowtheory.com/huygens/huygens/doc/index.html).

See also
--------

Some related projects:

[Foundations of brick diagrams](https://arxiv.org/abs/1908.10660)
Describes horizontal and vertical composition of string diagrams.

[diagrams](https://archives.haskell.org/projects.haskell.org/diagrams/)
A declarative, domain-specific language written in Haskell.

[Compose.jl](https://github.com/GiovineItalia/Compose.jl)
A declarative vector graphics library for Julia.

[Experiments in Constraint-based Graphic Design](https://www.anishathalye.com/2019/12/12/constraint-based-graphic-design/#case-studies) 
Describes a DSL called "Basalt" written in Python, based on
the logic of non-linear real arithmetic, which is an
extension of linear programming. No code available.

[Penrose](http://penrose.ink) 
"Create beautiful diagrams just by typing mathematical notation in plain text."
[Source here](https://github.com/penrose/penrose)
Written in Haskell.  Users not welcome (yet).

[Graphviz](https://graphviz.org/)
Declarative language for rendering heirarchical data structures.
[Apparently](https://news.ycombinator.com/item?id=23477034)
"20 year old code that was basically a prototype that escaped from the lab".
See also 
[this blog post.](https://ncona.com/2020/06/create-diagrams-with-code-using-graphviz/)


[TikZ](https://ctan.org/pkg/pgf?lang=en)
Possibly the most widely used package for creating
graphics for integration with latex documents.



