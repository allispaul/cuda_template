CXX=nvcc
APP=main

CXXFLAGS=--generate-code=arch=compute_89,code=sm_89 -std=c++17 -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --expt-relaxed-constexpr -lineinfo

LDFLAGS=

LDLIBS=-lcuda

OBJECTS = ${APP}.o

.SUFFIXES: .o .cu

default: clean ${APP}

${APP}: $(OBJECTS)
        $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

${APP}.o:
        $(CXX) -c $(CXXFLAGS) -o "$@" ${APP}.cu

clean:
        rm -f $(OBJECTS) ${APP}
