CC=gcc
CFLAGS=-O3 -std=gnu11 -Wall -Wextra
EXEC=recpart hi2vec renum recpart_attr hi2vec_attr

all: $(EXEC)

recpart: partition.o attr.o recpart.o
	$(CC) -o recpart partition.o attr.o recpart.o $(CFLAGS) -lm

recpart_attr: partition.o attr.o recpart_attr.o
	$(CC) -o recpart_attr partition.o attr.o recpart_attr.o $(CFLAGS) -lm

hi2vec: hi2vec.c
	$(CC) -o hi2vec hi2vec.c $(CFLAGS) -lm

hi2vec_attr: hi2vec_attr.c attr.o
	$(CC) -o hi2vec_attr hi2vec_attr.c attr.o $(CFLAGS) -lm

renum: renum.c
	$(CC) -o renum renum.c $(CFLAGS)

clean:
	rm -f *.o recpart hi2vec renum recpart_attr hi2vec_attr

%.o: %.c %.h
	$(CC) -o $@ -c $< $(CFLAGS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)
