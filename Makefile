APP=main
ARCH=90a
CUTLASS_DIR=~/Documents/cuda/cutlass/

CXX=nvcc
CXXFLAGS=--generate-code=arch=compute_${ARCH},code=sm_${ARCH} -std=c++17 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --expt-relaxed-constexpr -Xptxas=-v,--warn-on-spills -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/tools/util/include

NDEBUGFLAGS=-DNDEBUG -lineinfo -O3

DEBUGFLAGS=-G -g

LDFLAGS=

LDLIBS=-lcuda

default: ${APP}

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
