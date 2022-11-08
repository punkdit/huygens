

canvas.html: canvas.ipynb
	jupyter nbconvert --to html canvas.ipynb 

canvas.ipynb: canvas.py
	./make_nb.py convert_to_nb  canvas.py canvas.ipynb

turtle.html: turtle.ipynb
	jupyter nbconvert --to html turtle.ipynb 

turtle.ipynb: turtle.py
	./make_nb.py convert_to_nb  turtle.py turtle.ipynb

wiggle.html: wiggle.ipynb
	jupyter nbconvert --to html wiggle.ipynb 

wiggle.ipynb: wiggle.py
	./make_nb.py convert_to_nb  wiggle.py wiggle.ipynb


