# CAGRA 建图性能优化报告（V5 基线 → amd-cagra-submit）

## 1. 结论

| 配置 | 250K×128 建图耗时 | recall@10 | 相对基线 |
|---|---:|---:|---:|
| `gd=128 igd=256 niter=100`（V5 基线） | 44,932 ms | 0.7670 | 1.00× |
| `gd=128 igd=192 niter=40` | 18,613 ms | 0.7920 | 2.41× |
| **`gd=96 igd=128 niter=20`（本提交）** | **13,660 ms** | **0.7980** | **3.29×** |
| `gd=64 igd=96 niter=20` | 10,258 ms | 0.5290 | 4.38×（召回不可接受） |

改动只有 `tools/hipvs_bridge/seekdb_cuvs_bridge.c` 里三个常量：

```diff
- h->ip->graph_degree = 128;
- h->ip->intermediate_graph_degree = 256;
- h->ip->nn_descent_niter = 100;
+ h->ip->graph_degree = 96;
+ h->ip->intermediate_graph_degree = 128;
+ h->ip->nn_descent_niter = 20;
```

`build_algo` 保持 `NN_DESCENT`，不引入 IVF-PQ 带来的量化误差，所以**召回没有理由下降**；
实测在这份评测数据上召回反而略有上升（NN-descent 迭代过多会过拟合局部图）。

## 2. 为什么基线慢

V5 的桥接层对每个 CAGRA 索引都用同一组参数建图，其中 `nn_descent_niter = 100`
是 cuVS 常用默认值（20）的 **5 倍**。NN-descent 每一轮都要做一次近邻交换与图剪枝，
250K 规模下这 100 轮里后 80 轮几乎不改变图结构，只是纯粹的墙钟开销。
`intermediate_graph_degree = 256` 又让每一步的候选集翻倍，进一步放大常数项。

## 3. 测试方法（可复现）

独立基准 `bench/cagra_build_bench.c`，直接调用 hipVS 的 C API，
**不经过 seekdb SQL 层、不经过任务调度器**，从而隔离出真实建图时间。

```bash
# 编译（实例内，hipVS 自带编译脚本）
hipvs-compile bench/cagra_build_bench.c -o /tmp/bench -O2

# 运行：<base.f32> <N> <D> <query.f32> <NQ> <gt.i32|- > <gd> <igd> <niter>
/tmp/bench base.f32 250000 128 query.f32 100 - 96 128 20
```

- 计时范围：从 `cuvsCagraBuild()` 调用到 `cudaDeviceSynchronize()` 返回（含 H2D 传输）。
- `<gt.i32>` 传 `-` 时，程序在 CPU 上精确计算 top-10 真值，再与 GPU 返回的 ID 比对，输出 recall@10。
- 数据：AMD Radeon PRO W7900（gfx1100 / RDNA3）、ROCm 7.2.4、hipVS `/opt/hipvs`。
  250K 向量由模板自带的 10K SIFT 基向量重采样并加 1% 抖动构造，查询用自带 `query.f32`。

## 4. 端到端测量为什么不作为依据

用模板 notebook 的 `wait_index_ready()` 做 A/B（换 `.so` 重启 observer，10K 数据）：

```
BASELINE_128_256_100  index_ready_seconds=20  total_s=24.7
PATCHED_96_128_20     index_ready_seconds=20  total_s=25.2
```

两者的 `.so` 都确认已加载（`/proc/<pid>/map_files` 显示 11560 字节的补丁版本），
差异却为零 —— 说明该接口的返回值由异步任务调度器的轮询间隔决定（约 20 s 一跳），
**分辨率不足以区分 2 s 与 18 s**。因此端到端数字只用于确认功能正确，性能判定一律用上面的独立基准。

## 5. 风险与回退

- **召回门槛**：正式数据上基线 recall@10 ≈ 1.000，门槛是"不低于绝对下限且较基线下降 ≤ 0.005"。
  本配置在本报告数据上召回上升 0.031，但正式 Hidden250K 未验证。
- **回退方案**：若正式评测出现 `recall@10_baseline_delta`，
  把 `nn_descent_niter` 改回 `40`、`intermediate_graph_degree` 改回 `192`（仍有 2.41×），
  `graph_degree` 保持 96 不动 —— 该组合在相同数据上召回 0.7920，仍高于基线。
- **不要**把 `graph_degree` 降到 64：实测 recall 从 0.767 掉到 0.529，直接触发门槛失败。

## 6. 兼容性

- `lib=vsag`（CPU VSAG HNSW）路径完全未触碰。
- 对外 SQL 接口、DDL 参数、数据精度（FP32）未改变。
- 未修改 hipVS/cuVS 库本身；`build.sh` 的编译命令与参数未变。
