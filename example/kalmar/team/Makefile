KOKKOS_PATH = /tmp/ctk/kokkos
SRC = main.cpp

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
KBASE=/opt/hcc
CXX = $(KBASE)/bin/clang++
CXXFLAGS = -O3
LINK = $(KBASE)/bin/clang++
LINKFLAGS = `$(KBASE)/bin/hcc-config --install --ldflags`
EXE = $(SRC:.cpp=.host)
KOKKOS_DEVICES = "Serial,Kalmar"
KOKKOS_ARCH = 
endif

CXXFLAGS += `$(KBASE)/bin/hcc-config --install --cxxflags`
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



