# C++ Compiler and options
CPP=g++ -O3 -march=native

# Path to hogwild (e.g. hogwildtl/include)
HOG_INCL=hogwildtl/include
# Path to Hazy Template Library (e.g. hazytl/include)
HTL_INCL=hazytl/include

LIBS=-lpthread -lnuma

# Conversion tools
TOOLS=bin/convert_matlab bin/convert bin/unconvert
UNAME=$(shell uname)
ifneq ($(UNAME), Darwin)
	LIB_RT=-lrt
endif

ALL= $(TOOLS) obj/frontend.o bin/svm bin/numasvm bin/mysvm

all: $(ALL)

obj/frontend.o:
	$(CPP) -c src/frontend_util.cc -o obj/frontend.o

bin/svm: obj/frontend.o
	$(CPP) -o bin/svm src/svm_main.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/numasvm: obj/frontend.o
	$(CPP) -o bin/numasvm src/numasvm_main.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/mysvm: obj/frontend.o
	$(CPP) -o bin/mysvm src/mynumasvm_main.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/bbsvm: obj/frontend.o
	$(CPP) -o bin/bbsvm src/bbsvm_main.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/tracenorm: obj/frontend.o
	$(CPP) -o bin/tracenorm src/tracenorm.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/predict: 
	$(CPP) -o bin/predict src/tracenorm/predict.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/bbtracenorm: obj/frontend.o
	$(CPP) -o bin/bbtracenorm src/bbtracenorm.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/multicut: obj/frontend.o
	$(CPP) -o bin/multicut src/multicut.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/bbmulticut: obj/frontend.o
	$(CPP) -o bin/bbmulticut src/bbmulticut.cc -I$(HOG_INCL) -I$(HTL_INCL) $(LIBS) $(LIB_RT) \
		obj/frontend.o

bin/convert: src/tools/tobinary.cc
	$(CPP) -o bin/convert src/tools/tobinary.cc -I$(HOG_INCL) -I$(HTL_INCL) 

bin/convert_matlab: src/tools/tobinary.cc
	$(CPP) -o bin/convert_matlab src/tools/tobinary.cc -I$(HOG_INCL) -I$(HTL_INCL) -DMATLAB_CONVERT_OFFSET=1

bin/unconvert: src/tools/unconvert.cc
	$(CPP) -o bin/unconvert src/tools/unconvert.cc -I$(HOG_INCL) -I$(HTL_INCL) 

clean:
	rm -f $(ALL)

datasets: data/news20_train.tsv data/rcv1_train.tsv data/rcv1_test.tsv # data/epsilon_test.tsv data/epsilon_train.tsv data/webspam_train.tsv

data/rcv1_test.tsv:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
	bunzip2 rcv1_train.binary.bz2
	python3 convert2hogwild.py rcv1_train.binary data/rcv1_test.tsv && rm rcv1_train.binary

data/rcv1_train.tsv:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
	bunzip2 rcv1_test.binary.bz2
	python3 convert2hogwild.py rcv1_test.binary data/rcv1_train.tsv && rm rcv1_test.binary

data/news20_train.tsv:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
	bunzip2 news20.binary.bz2
	python3 convert2hogwild.py news20.binary data/news20 --split && rm news20.binary

data/epsilon_test.tsv:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.xz
	unxz epsilon_normalized.t.xz
	python3 convert2hogwild.py epsilon_normalized.t data/epsilon_test.tsv && rm epsilon_normalized.t

data/epsilon_train.tsv:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.xz
	unxz epsilon_normalized.xz
	python3 convert2hogwild.py epsilon_normalized data/epsilon_train.tsv && rm epsilon_normalized

data/webspam_train.tsv:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.xz
	unxz webspam_wc_normalized_trigram.svm.xz
	python3 convert2hogwild.py webspam_wc_normalized_trigram.svm data/webspam --split && rm webspam_wc_normalized_trigram.svm

