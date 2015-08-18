KOKKOS_PATH = ${HOME}/kokkos
SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"
	
ifeq ($(CUDA), yes)
CXX = nvcc_wrapper
CXXFLAGS = -O3
LINK = nvcc_wrapper
LINKFLAGS = 
EXE = $(SRC:.cpp=.cuda)
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = /opt/kalmar/bin/clang++
CXXFLAGS = -O3
LINK = /opt/kalmar/bin/clang++
LINKFLAGS = -hc -L/opt/kalmar/lib -Wl,--rpath=/opt/kalmar/lib -lc++ -lcxxrt -ldl -lpthread -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive  
EXE = $(SRC:.cpp=.host)
KOKKOS_DEVICES = "Kalmar,Serial"
KOKKOS_ARCH = 
endif

CXXFLAGS += -hc -I/opt/kalmar/include -I/opt/kalmar/include/c++/v1 
LDFLAGS += 

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<



