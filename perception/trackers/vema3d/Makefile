CXX ?= g++
DS_INC ?= /opt/nvidia/deepstream/deepstream/sources/includes

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -fPIC -Iinclude -I$(DS_INC)
LDFLAGS := -shared -pthread

TARGET := libnvds_vema3d_tracker.so
SRCS := src/vema3d_tracker.cpp src/nvmot_context.cpp
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

%.o: %.cpp include/vema3d_tracker.hpp Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -f $(OBJS) $(TARGET)
