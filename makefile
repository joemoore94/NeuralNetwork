all: RunNet

RunNet: main.o network.o readfile.o sigmoid.o softmax.o
	g++ -std=c++14 main.o network.o readfile.o sigmoid.o softmax.o -o RunNet

main.o: main.cpp
	g++ -std=c++14 -Wall -c main.cpp

network.o: network.hpp network.cpp
	g++ -std=c++14 -Wall -c network.cpp

readfile.0: readfile.hpp readfile.cpp
	g++ -std=c++14 -Wall -c readfile.cpp

sigmoid.o: sigmoid.hpp sigmoid.cpp
	g++ -std=c++14 -Wall -c sigmoid.cpp

softmax.o: softmax.hpp softmax.cpp
	g++ -std=c++14 -Wall -c softmax.cpp

clean:
	rm *.o RunNet
