CXX=nvcc
APP=main

CXXFLAGS=--generate-code=arch=compute_89,code=sm_89 -std=c++17 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --expt-relaxed-constexpr

NDEBUGFLAGS=-DNDEBUG -lineinfo -O3

DEBUGFLAGS=-G -g

LDFLAGS=

LDLIBS=-lcuda

default: clean ${APP}

debug: ${APP}.cu
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(LDFLAGS) -o ${APP} $(LDLIBS) ${APP}.cu

${APP}: ${APP}.cu
	$(CXX) $(CXXFLAGS) $(NDEBUGFLAGS) $(LDFLAGS) -o ${APP} $(LDLIBS) ${APP}.cu

${APP}.ptx: ${APP}.cu
	$(CXX) $(CXXFLAGS) $(NDEBUGFLAGS) -ptx -o $@ ${APP}.cu

${APP}.cubin: ${APP}.cu
	$(CXX) $(CXXFLAGS) $(NDEBUGFLAGS) -cubin -o $@ ${APP}.cu

${APP}.sass: ${APP}.cubin
	nvdisasm ${APP}.cubin > $@

clean:
	rm -f ${APP} ${APP}.sass ${APP}.ptx ${APP}.cubin
