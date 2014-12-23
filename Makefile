
all: nlpnet/network.c
	python setup.py build

nlpnet/network.c: nlpnet/network.pyx nlpnet/networklm.pyx nlpnet/networkSent.pyx
	cython $<
