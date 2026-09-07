# VectorAdd CPU vs GPU Timing Results

| N (2^e) | N (elements) | CPU time (ms) | GPU time (ms) |
|---|---|---|---|
| 1<<24 | 16777216 | 37.782500 | 31.679935 |
| 1<<28 | 268435456 | 578.076700 | 476.421509 |

## Notes

### Timing

- CUDA is asynchronous: a kernel launch (`vector_add<<<...>>>(...)`) returns immediately on the CPU. The CPU keeps executing the next line of code without waiting for the GPU to actually finish the work.
- `cudaEvent_t` timing functions (`cudaEventCreate`, `cudaEventRecord`) are CPU-side API calls. `cudaEventRecord` doesn't measure time itself — it enqueues a timestamp marker into the GPU's stream, to be filled in whenever the GPU actually reaches that point.
- `cudaEventSynchronize()` blocks the CPU until the GPU has reached (and recorded) that event. This is why it has to be called before `cudaEventElapsedTime()` — otherwise you'd be asking for the time difference between two markers, one or both of which the GPU may not have reached yet.

**Timing CPU work:** use the standard time library (`std::chrono`) — take a timestamp before and after the code, then subtract.

**Timing GPU work:** can't just wrap the kernel launch in `chrono` timestamps, because the launch returns before the GPU is done (see async note above). Instead:
1. `cudaEventRecord(start)` — enqueue a start marker
2. run the GPU work
3. `cudaEventRecord(stop)` — enqueue a stop marker
4. `cudaEventSynchronize(stop)` — wait for the GPU to actually reach the stop marker
5. `cudaEventElapsedTime(&ms, start, stop)` — read the time between the two markers

### GPU programming model

- The CPU (host) runs the normal program flow and also launches kernels that run on the GPU (device).
- The GPU's speed advantage comes from data parallelism — the same operation applied independently across many elements at once (e.g. one thread per array index in vector add), not from being a faster single core.

**Step-by-step flow for offloading work to the GPU:**
1. Move data to the GPU
   - Allocate GPU memory (`cudaMalloc`)
   - Copy inputs (A and B) from CPU to GPU (`cudaMemcpy`, `cudaMemcpyHostToDevice`)
2. Launch the kernel — runs across many device threads in parallel
3. Get the result back — copy the output from GPU to CPU (`cudaMemcpy`, `cudaMemcpyDeviceToHost`)

**Gotcha:** when specifying a size in bytes for `cudaMalloc`/`cudaMemcpy`, use `sizeof(the element type)`, not `sizeof(the pointer)`. `sizeof(float*)` is always the pointer's size (e.g. 8 bytes on a 64-bit system) regardless of how many elements it points to — you want `N * sizeof(float)`.
