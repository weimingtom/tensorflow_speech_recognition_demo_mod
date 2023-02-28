PIP := pip
PYTHON := python
RM := rm -rf

all:
	$(PIP) install tensorflow  
	$(PIP) install librosa  
	$(PIP) install tflearn
	$(PIP) install scikit-image  
	$(PYTHON) demo.py  

test:
	$(PYTHON) test.py  

clean:
	$(RM) tflearn.lstm.model.* checkpoint
	$(RM) data/spoken_numbers_pcm/
	$(RM) __pycache__ .ipynb_checkpoints


