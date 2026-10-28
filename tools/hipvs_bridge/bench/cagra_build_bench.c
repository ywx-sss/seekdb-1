/*
 * cagra_build_bench.c -- standalone CAGRA build-time / recall@10 benchmark.
 *
 * Calls the hipVS (cuVS) C API directly so that the number reported is the
 * graph-construction cost alone: no seekdb SQL layer, no async task
 * scheduler, no snapshot serialization.
 *
 * Build (inside an instance that has /opt/hipvs, e.g. the seekdb-hipvs image):
 *     hipvs-compile cagra_build_bench.c -o bench -O2
 *
 * Run:
 *     ./bench <base.f32> <N> <D> <query.f32> <NQ> <gt.i32|-> <gd> <igd> <niter>
 *
 *   gt.i32   ground-truth ids, NQ x 10 int32, row major. Pass "-" to compute
 *            exact top-10 on the CPU instead (slower, but self-contained).
 *
 * Output (single line, machine readable):
 *     RESULT N=... D=... NQ=... gd=... igd=... niter=... build_ms=... recall10=...
 */
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include <cuvs/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CKH(x) do { hipError_t _h = (x); if (_h != hipSuccess) { \
  fprintf(stderr, "HIP FAIL %d\n", __LINE__); exit(1); } } while (0)
#define CKC(x) do { cuvsError_t _e = (x); if (_e != CUVS_SUCCESS) { \
  fprintf(stderr, "CUVS FAIL %d: %s\n", __LINE__, cuvsGetLastErrorText()); exit(1); } } while (0)

static double now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void *read_file(const char *path, size_t bytes)
{
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  void *buf = malloc(bytes);
  if (fread(buf, 1, bytes, f) != bytes) {
    fprintf(stderr, "short read on %s\n", path);
    exit(1);
  }
  fclose(f);
  return buf;
}

int main(int argc, char **argv)
{
  if (argc < 10) {
    fprintf(stderr, "usage: %s base.f32 N D query.f32 NQ gt.i32|- gd igd niter\n", argv[0]);
    return 2;
  }
  const char *base_path = argv[1];
  long        N         = atol(argv[2]);
  long        D         = atol(argv[3]);
  const char *query_path = argv[4];
  long        NQ        = atol(argv[5]);
  const char *gt_path   = argv[6];
  long        GD        = atol(argv[7]);
  long        IGD       = atol(argv[8]);
  long        NITER     = atol(argv[9]);

  float *data = (float *)read_file(base_path, (size_t)N * D * sizeof(float));
  float *qry  = (float *)read_file(query_path, (size_t)NQ * D * sizeof(float));

  cuvsResources_t res;
  CKC(cuvsResourcesCreate(&res));

  DLManagedTensor dset;
  memset(&dset, 0, sizeof(dset));
  long ds[2] = {N, D};
  dset.dl_tensor.data                 = data;
  dset.dl_tensor.device.device_type   = kDLCPU;
  dset.dl_tensor.device.device_id     = 0;
  dset.dl_tensor.ndim                 = 2;
  dset.dl_tensor.dtype.code           = kDLFloat;
  dset.dl_tensor.dtype.bits           = 32;
  dset.dl_tensor.dtype.lanes          = 1;
  dset.dl_tensor.shape                = ds;
  dset.dl_tensor.strides              = 0;
  dset.dl_tensor.byte_offset          = 0;

  cuvsCagraIndexParams_t ip;
  CKC(cuvsCagraIndexParamsCreate(&ip));
  ip->build_algo                = NN_DESCENT;
  ip->graph_degree              = (size_t)GD;
  ip->intermediate_graph_degree = (size_t)IGD;
  ip->nn_descent_niter          = (size_t)NITER;

  cuvsCagraIndex_t idx;
  CKC(cuvsCagraIndexCreate(&idx));

  double t0 = now_ms();
  CKC(cuvsCagraBuild(res, ip, &dset, idx));
  CKH(cudaDeviceSynchronize());
  double t1 = now_ms();

  /* ---- one top-10 search so the index is actually usable ---- */
  float        *qd = 0;
  unsigned int *nd = 0;
  float        *dd = 0;
  CKC(cuvsRMMAlloc(res, (void **)&qd, (size_t)NQ * D * sizeof(float)));
  CKC(cuvsRMMAlloc(res, (void **)&nd, (size_t)NQ * 10 * sizeof(unsigned int)));
  CKC(cuvsRMMAlloc(res, (void **)&dd, (size_t)NQ * 10 * sizeof(float)));
  CKH(cudaMemcpy(qd, qry, (size_t)NQ * D * sizeof(float), cudaMemcpyDefault));

  DLManagedTensor qt, nt, dt;
  memset(&qt, 0, sizeof(qt));
  long qs[2] = {NQ, D}, ns[2] = {NQ, 10}, ds2[2] = {NQ, 10};
  qt.dl_tensor.data = qd; qt.dl_tensor.device.device_type = kDLCUDA;
  qt.dl_tensor.ndim = 2;  qt.dl_tensor.dtype.code = kDLFloat;
  qt.dl_tensor.dtype.bits = 32; qt.dl_tensor.dtype.lanes = 1; qt.dl_tensor.shape = qs;
  nt = qt; nt.dl_tensor.data = nd; nt.dl_tensor.dtype.code = kDLUInt; nt.dl_tensor.shape = ns;
  dt = qt; dt.dl_tensor.data = dd; dt.dl_tensor.shape = ds2;

  cuvsCagraSearchParams_t sp;
  CKC(cuvsCagraSearchParamsCreate(&sp));
  {
    size_t itopk = (size_t)10 * 4;
    if (itopk < 512) itopk = 512;
    if (sp->itopk_size < itopk) sp->itopk_size = itopk;
    sp->search_width = 4;
  }
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)0;
  CKC(cuvsCagraSearch(res, sp, idx, &qt, &nt, &dt, filter));
  CKH(cudaDeviceSynchronize());

  unsigned int *ids = (unsigned int *)malloc((size_t)NQ * 10 * sizeof(unsigned int));
  CKH(cudaMemcpy(ids, nd, (size_t)NQ * 10 * sizeof(unsigned int), cudaMemcpyDefault));

  /* ---- recall@10 ---- */
  long hit = 0, total = 0;
  if (strcmp(gt_path, "-") != 0) {
    int *gt = (int *)read_file(gt_path, (size_t)NQ * 10 * sizeof(int));
    for (long q = 0; q < NQ; q++) {
      for (int j = 0; j < 10; j++) {
        total++;
        for (int m = 0; m < 10; m++) {
          if ((int)ids[q * 10 + m] == gt[q * 10 + j]) { hit++; break; }
        }
      }
    }
  } else {
    for (long q = 0; q < NQ; q++) {
      float bv[10];
      long  bi[10];
      for (int j = 0; j < 10; j++) { bv[j] = 1e30f; bi[j] = -1; }
      const float *qv = qry + q * D;
      for (long i = 0; i < N; i++) {
        const float *dv = data + i * D;
        float d = 0;
        for (long k = 0; k < D; k++) { float t = qv[k] - dv[k]; d += t * t; }
        if (d < bv[9]) {
          int j = 9;
          while (j > 0 && d < bv[j - 1]) { bv[j] = bv[j - 1]; bi[j] = bi[j - 1]; j--; }
          bv[j] = d; bi[j] = i;
        }
      }
      for (int j = 0; j < 10; j++) {
        total++;
        for (int m = 0; m < 10; m++) {
          if ((long)ids[q * 10 + m] == bi[j]) { hit++; break; }
        }
      }
    }
  }

  printf("RESULT N=%ld D=%ld NQ=%ld gd=%ld igd=%ld niter=%ld build_ms=%.1f recall10=%.4f\n",
         N, D, NQ, GD, IGD, NITER, t1 - t0, (double)hit / (double)total);
  return 0;
}
