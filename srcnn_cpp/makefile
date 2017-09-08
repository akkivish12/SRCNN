CC	= g++
CFLAGS = -std=c++11 -g `pkg-config --cflags opencv` 
TARGET = srcnn
SOURCES = srcnn.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

LIBS	= `pkg-config --libs opencv`

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) $(LIBS) 

.SUFFIXES: .cpp .o

.cpp.o:
	$(CC) -c $(CFLAGS) $<

.PHONY: clean

clean:
	/bin/rm -f core *.o $(OBJECTS) $(TARGET) 