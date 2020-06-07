all: RunNet

RunNet: main.o network.o readfile.o convolution.o maxpool.o
	g++ -std=c++14 main.o network.o readfile.o convolution.o maxpool.o -o RunNet

main.o: main.cpp network.hpp readfile.hpp convolution.hpp maxpool.hpp
	g++ -std=c++14 -Wall -c main.cpp

network.o: network.hpp network.cpp readfile.hpp convolution.hpp
	g++ -std=c++14 -Wall -c network.cpp

readfile.0: readfile.hpp readfile.cpp
	g++ -std=c++14 -Wall -c readfile.cpp

convolution.o: convolution.hpp convolution.cpp
	g++ -std=c++14 -Wall -c convolution.cpp

maxpool.o: maxpool.hpp maxpool.cpp
	g++ -std=c++14 -Wall -c maxpool.cpp

clean:
	rm *.o RunNet
