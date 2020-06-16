BASEDIR = ./
SRCDIR = $(BASEDIR)source/
EXE = CornerDetector
RUN = $(SRCDIR)$(EXE)
OUTPUT_DIR = $(BASEDIR)output/

all: 
	nvcc $(SRCDIR)CornerDetector.cu -o $(SRCDIR)$(EXE)

clean:
	rm $(RUN)
	rm $(OUTPUT_DIR)*

run: 
	cd	$(SRCDIR) && \
	$(BASEDIR)$(EXE) ../input/image1.ppm gaussMask=5 sigma=0.5 tpb=16 && \
    $(BASEDIR)$(EXE) ../input/image2.ppm gaussMask=7 tpb=32

