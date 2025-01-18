#include <cstdlib>
#include <iostream>
#include <string>


#include <cute/tensor.hpp>

#include <cutlass/util/command_line.h>
#include <cutlass/util/GPU_Clock.hpp>

////////////////////////////////////////////////////////////

#define CUDA_ERR_CHK(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

////////////////////////////////////////////////////////////

struct CLIArgs {
  std::string task {""};
  size_t kernel_num {0};
  size_t M {4096};
  size_t N {4096};
  size_t K {4096};
  size_t iters {20};
  size_t seed {0};

  std::ostream& print_usage(std::ostream &out) const {
    out << "APP_NAME\n\n"
        << "APP_DESCRIPTION\n\n"
        << "Options:\n\n"
        << "  --help                        If specified, displays this usage statement\n\n"
        << "  --task=[verify|profile|time]  Task to run\n"
        << "  --m=<int>,--n=<int>,--k=<int> Problem dimensions\n"
        << "  --seed=<int>                  Random seed\n"
        << "  --iters=<int>                 Iterations for timing\n"
        << "  --kernel=<int>                Which kernel to time/profile/verify\n"
        << "\n";
    return out;
  }

  void parse_args(int argc, char const* argv[]) {
    cutlass::CommandLine cli {argc, argv};
    if (cli.check_cmd_line_flag("help")) {
      print_usage(std::cout);
      exit(EXIT_FAILURE);
    }

    cli.get_cmd_line_argument("task", task);
    cli.get_cmd_line_argument("kernel", kernel_num);
    cli.get_cmd_line_argument("m", M);
    cli.get_cmd_line_argument("n", N);
    cli.get_cmd_line_argument("k", K);
    cli.get_cmd_line_argument("iters", iters);
    cli.get_cmd_line_argument("seed", seed);
  }
};

////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC>
void __global__ kernel(TA* A, TB* B, TC* C, int M, int N, int K) {
  // TODO
}

template <typename TA, typename TB, typename TC>
void __global__ kernel_ref(TA* A, TB* B, TC* C, int M, int N, int K) {
  // TODO
}

template <typename TA, typename TB, typename TC>
double dispatch(TA *A, TB *B, TC *C, int M, int N, int K, int iters, int kernel_num) {
  GPU_Clock timer;
  float secs;

  dim3 constexpr block_dim {1};
  dim3 grid_dim {1};

  switch (kernel_num) {
  case 0:
    timer.start();
    for (int i = 0; i < iters; ++i)
      kernel_ref<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    secs = timer.seconds();
    break;
  case 1:
    timer.start();
    for (int i = 0; i < iters; ++i)
      kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    secs = timer.seconds();
    break;
  default:
    std::cerr << "Unknown kernel number " << kernel_num << std::endl;
    exit(EXIT_FAILURE);
  }
  CUDA_ERR_CHK(cudaPeekAtLastError());
  CUDA_ERR_CHK(cudaDeviceSynchronize());
  return secs;
}

void profile(int M, int N, int K, int seed, int kernel_num) {
  using TA = float;
  using TB = float;
  using TC = float;

  int A_size = M * K;
  int B_size = N * K;
  int C_size = M * N;

  TA* hA = static_cast<TA*>(malloc(A_size * sizeof(TA)));
  TB* hB = static_cast<TB*>(malloc(B_size * sizeof(TB)));
  TC* hC = static_cast<TC*>(malloc(C_size * sizeof(TC)));

  if (hA == nullptr || hB == nullptr || hC == nullptr) {
    std::cerr << "malloc error\n";
    exit(EXIT_FAILURE);
  }

  srand(static_cast<unsigned int>(seed));
  for (int i = 0; i < A_size; ++i)
    hA[i] = static_cast<TA>(static_cast<double>(rand()) / RAND_MAX - 1);
  for (int i = 0; i < B_size; ++i)
    hB[i] = static_cast<TB>(static_cast<double>(rand()) / RAND_MAX - 1);
  for (int i = 0; i < C_size; ++i)
    hC[i] = static_cast<TC>(static_cast<double>(rand()) / RAND_MAX - 1);

  TA* dA;
  TB* dB;
  TC* dC;
  CUDA_ERR_CHK(cudaMalloc(&dA, A_size * sizeof(TA)));
  CUDA_ERR_CHK(cudaMalloc(&dB, B_size * sizeof(TB)));
  CUDA_ERR_CHK(cudaMalloc(&dC, C_size * sizeof(TC)));
  CUDA_ERR_CHK(cudaMemcpy(dA, hA, A_size * sizeof(TA), cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dB, hB, B_size * sizeof(TB), cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dC, hC, C_size * sizeof(TC), cudaMemcpyHostToDevice));

  dispatch(dA, dB, dC, M, N, K, 1, kernel_num);

  CUDA_ERR_CHK(cudaFree(dA));
  CUDA_ERR_CHK(cudaFree(dB));
  CUDA_ERR_CHK(cudaFree(dC));

  free(hA);
  free(hB);
  free(hC);
}

void verify(int M, int N, int K, int seed, int kernel_num) {
  using TA = float;
  using TB = float;
  using TC = float;

  int A_size = M * K;
  int B_size = N * K;
  int C_size = M * N;

  TA* hA = static_cast<TA*>(malloc(A_size * sizeof(TA)));
  TB* hB = static_cast<TB*>(malloc(B_size * sizeof(TB)));
  TC* hC = static_cast<TC*>(malloc(C_size * sizeof(TC)));
  TC* hCref = static_cast<TC*>(malloc(C_size * sizeof(TC)));

  if (hA == nullptr || hB == nullptr || hC == nullptr || hCref == nullptr) {
    std::cerr << "malloc error\n";
    exit(EXIT_FAILURE);
  }

  srand(static_cast<unsigned int>(seed));
  for (int i = 0; i < A_size; ++i)
    hA[i] = static_cast<TA>(static_cast<double>(rand()) / RAND_MAX - 1);
  for (int i = 0; i < B_size; ++i)
    hB[i] = static_cast<TB>(static_cast<double>(rand()) / RAND_MAX - 1);
  for (int i = 0; i < C_size; ++i)
    hC[i] = static_cast<TC>(static_cast<double>(rand()) / RAND_MAX - 1);
  for (int i = 0; i < C_size; ++i)
    hCref[i] = static_cast<TC>(static_cast<double>(rand()) / RAND_MAX - 1);

  TA* dA;
  TB* dB;
  TC* dC;
  TC* dCref;
  CUDA_ERR_CHK(cudaMalloc(&dA, A_size * sizeof(TA)));
  CUDA_ERR_CHK(cudaMalloc(&dB, B_size * sizeof(TB)));
  CUDA_ERR_CHK(cudaMalloc(&dC, C_size * sizeof(TC)));
  CUDA_ERR_CHK(cudaMalloc(&dCref, C_size * sizeof(TC)));
  CUDA_ERR_CHK(cudaMemcpy(dA, hA, A_size * sizeof(TA), cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dB, hB, B_size * sizeof(TB), cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dC, hC, C_size * sizeof(TC), cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dCref, hCref, C_size * sizeof(TC), cudaMemcpyHostToDevice));

  dispatch(dA, dB, dC, M, N, K, 1, kernel_num);
  dispatch(dA, dB, dCref, M, N, K, 1, kernel_num);

  CUDA_ERR_CHK(cudaMemcpy(hC, dC, C_size * sizeof(TC), cudaMemcpyDeviceToHost));
  CUDA_ERR_CHK(cudaMemcpy(hCref, dCref, C_size * sizeof(TC), cudaMemcpyDeviceToHost));

  TC max_diff = 0.0f;
  TC curr_diff;
  size_t curr_idx = 0;
  for (size_t i = 0; i < C_size; ++i) {
    curr_diff = std::abs(hCref[i] - hC[i]);
    if (curr_diff > max_diff) {
      curr_idx = i;
      max_diff = curr_diff;
    }
  }
  std::cout << "Maximum difference: " << max_diff << " at index " << curr_idx << "\n";
  std::cout << "Reference = " << hCref[curr_idx] << ", result = " << hC[curr_idx] << "\n";
  std::cout << "First 10 elements:\n";
  std::cout << "Reference: ";
  for (int i = 0; i < 10; ++i)
    std::cout << hCref[i] << "\t";
  std::cout << "\nResult: ";
  for (int i = 0; i < 10; ++i)
    std::cout << hC[i] << "\t";
  std::cout << "\n";

  CUDA_ERR_CHK(cudaFree(dA));
  CUDA_ERR_CHK(cudaFree(dB));
  CUDA_ERR_CHK(cudaFree(dC));
  CUDA_ERR_CHK(cudaFree(dCref));

  free(hA);
  free(hB);
  free(hC);
  free(hCref);
}

void time(int M, int N, int K, int iters, int seed, int kernel_num) {
  int constexpr warmup_iters = 5;

  using TA = float;
  using TB = float;
  using TC = float;

  int A_size = M * K;
  int B_size = N * K;
  int C_size = M * N;

  TA* hA = static_cast<TA*>(malloc(A_size * sizeof(TA)));
  TB* hB = static_cast<TB*>(malloc(B_size * sizeof(TB)));
  TC* hC = static_cast<TC*>(malloc(C_size * sizeof(TC)));

  if (hA == nullptr || hB == nullptr || hC == nullptr) {
    std::cerr << "malloc error\n";
    exit(EXIT_FAILURE);
  }

  srand(static_cast<unsigned int>(seed));
  for (int i = 0; i < A_size; ++i)
    hA[i] = static_cast<TA>(static_cast<double>(rand()) / RAND_MAX - 1);
  for (int i = 0; i < B_size; ++i)
    hB[i] = static_cast<TB>(static_cast<double>(rand()) / RAND_MAX - 1);
  for (int i = 0; i < C_size; ++i)
    hC[i] = static_cast<TC>(static_cast<double>(rand()) / RAND_MAX - 1);

  TA* dA;
  TB* dB;
  TC* dC;
  CUDA_ERR_CHK(cudaMalloc(&dA, A_size * sizeof(TA)));
  CUDA_ERR_CHK(cudaMalloc(&dB, B_size * sizeof(TB)));
  CUDA_ERR_CHK(cudaMalloc(&dC, C_size * sizeof(TC)));
  CUDA_ERR_CHK(cudaMemcpy(dA, hA, A_size * sizeof(TA), cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dB, hB, B_size * sizeof(TB), cudaMemcpyHostToDevice));
  CUDA_ERR_CHK(cudaMemcpy(dC, hC, C_size * sizeof(TC), cudaMemcpyHostToDevice));


  dispatch(dA, dB, dC, M, N, K, warmup_iters, kernel_num);
  double secs = dispatch(dA, dB, dC, M, N, K, iters, kernel_num);

  std::cout << "Kernel " << kernel_num << ": " << secs * 1000000.0 << "ms\n";

  CUDA_ERR_CHK(cudaFree(dA));
  CUDA_ERR_CHK(cudaFree(dB));
  CUDA_ERR_CHK(cudaFree(dC));

  free(hA);
  free(hB);
  free(hC);
}

int main(int argc, char const* argv[]) {
  CLIArgs options;
  options.parse_args(argc, argv);

  if (options.task == "profile")
    profile(options.M, options.N, options.K, options.seed, options.kernel_num);
  else if (options.task == "verify")
    verify(options.M, options.N, options.K, options.seed, options.kernel_num);
  else if (options.task == "time")
    time(options.M, options.N, options.K, options.iters, options.seed, options.kernel_num);
  else {
    std::cerr << "Unknown task " << options.task << "\n";
    options.print_usage(std::cerr);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
