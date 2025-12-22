# Understanding Key Features: CUDA Graph, Tensor Parallelism, and Radix Cache

This document provides a detailed walkthrough of three critical features in Mini-SGLang.

## 1. CUDA Graph Capture and Replay

### Overview
CUDA graphs minimize CPU launch overhead during the decode phase by pre-capturing GPU operations and replaying them. This is especially important for small batch sizes where CPU overhead can dominate.

### Motivation: Why Capture and Replay CUDA Graphs?

#### The Fundamental Problem

**LLM inference, especially the decode phase, is highly repetitive:**
- Same sequence of operations: embedding → transformer layers → LM head
- Same kernel launches: attention, matrix multiplications, activations
- Same memory access patterns: KV cache reads, weight accesses
- **This repetition happens thousands of times per second** during serving

**The CPU Launch Overhead Bottleneck:**

In traditional GPU execution, each kernel launch requires:
1. **CPU-GPU Communication**: CPU sends launch command to GPU driver
2. **Driver Processing**: GPU driver processes the launch request
3. **Kernel Scheduling**: GPU scheduler queues the kernel
4. **Synchronization**: CPU may wait for completion before launching next kernel

For a typical LLM forward pass with 32 layers:
- **Normal execution**: ~64-128 kernel launches (2-4 per layer)
- **Each launch**: ~5-20μs CPU overhead
- **Total CPU overhead**: ~320-2560μs (0.3-2.6ms) per forward pass
- **For batch_size=1 decode**: CPU overhead can be **50-80% of total latency**

**The Repetition Problem:**

During decode phase, we generate tokens one at a time:
```
Token 1: Forward pass (64 kernel launches, ~500μs GPU + ~500μs CPU overhead)
Token 2: Forward pass (64 kernel launches, ~500μs GPU + ~500μs CPU overhead)
Token 3: Forward pass (64 kernel launches, ~500μs GPU + ~500μs CPU overhead)
...
Token 1000: Forward pass (64 kernel launches, ~500μs GPU + ~500μs CPU overhead)
```

**The same 64 kernels are launched 1000 times!** The CPU is doing redundant work:
- Re-evaluating which kernels to launch (even though it's always the same)
- Re-computing launch parameters (even though they're predictable)
- Re-synchronizing between kernels (even though the sequence is fixed)

#### The Solution: Capture Once, Replay Many Times

**CUDA Graph Capture:**
- **Record once**: During initialization, capture the entire forward pass as a graph
- **Optimize once**: CUDA optimizes the graph structure, kernel scheduling, memory access
- **Store once**: Save the optimized execution plan

**Where Optimization Happens (In Code):**
```python
# Location: python/minisgl/engine/graph.py:144-145

with torch.cuda.graph(g, pool=pool, stream=stream):
    self.logits[:bs] = model.forward()  # CUDA records and optimizes here
```

**Inside the `torch.cuda.graph()` context manager:**
1. **Recording Phase**: CUDA records all GPU operations (kernels, memory transfers) executed inside the context
2. **Optimization Phase**: CUDA runtime automatically optimizes:
   - **Kernel scheduling**: Determines optimal execution order and overlap
   - **Memory access patterns**: Optimizes memory coalescing and prefetching
   - **Kernel fusion opportunities**: Identifies operations that can be merged
   - **Register allocation**: Optimizes register usage across kernels
   - **Pipeline optimization**: Overlaps computation and memory transfers
3. **Storage**: The optimized execution plan is stored **inside the `g` object** (a `torch.cuda.CUDAGraph` instance)

**Where the Optimized Plan is Stored (In Code):**
```python
# Location: python/minisgl/engine/graph.py:138, 148, 153

g = torch.cuda.CUDAGraph()  # Create graph object
# ... capture happens ...
graph_list.append((bs, g))  # Store graph object
self.graph_map = dict(graph_list)  # Map batch_size → optimized graph
```

**Storage Details:**
- Each `torch.cuda.CUDAGraph` object (`g`) contains the optimized execution plan internally
- The plan is stored in GPU memory and managed by CUDA runtime
- `self.graph_map` (line 153) maps batch sizes to their corresponding optimized graphs
- During replay, we retrieve the graph: `g = self.graph_map[batch.padded_size]` (line 163)

**Note**: The optimization is done **automatically by CUDA runtime/driver** - PyTorch code doesn't explicitly call optimization functions. The optimization happens transparently when exiting the `torch.cuda.graph()` context manager.

**CUDA Graph Replay:**
- **Launch once**: Single CPU command launches the entire graph
- **Execute efficiently**: GPU executes all kernels back-to-back without CPU intervention
- **Repeat efficiently**: Replay the same optimized graph thousands of times

**The Key Insight:**

Since decode phase is **predictable and repetitive**, we can:
1. **Pay the capture cost once** (during initialization, ~100-500ms)
2. **Amortize over thousands of replays** (each replay saves ~200-500μs)
3. **Break-even point**: After ~200-2500 replays, we've saved more time than capture cost
4. **In production**: Millions of replays → massive time savings

#### Why This Matters for LLM Serving

**1. Latency Reduction:**
- **User-facing**: Lower latency = better user experience
- **Interactive applications**: Chat, code completion, real-time generation
- **Time-to-first-token**: Critical for streaming responses

**2. Throughput Improvement:**
- **More requests per second**: CPU can handle more concurrent requests
- **Better GPU utilization**: Less CPU waiting = more GPU computation
- **Scalability**: System can serve more users with same hardware

**3. Cost Efficiency:**
- **Lower infrastructure costs**: Same hardware handles more load
- **Energy efficiency**: Less CPU-GPU communication = lower power consumption
- **Resource optimization**: Better utilization of expensive GPU resources

**4. Production Reliability:**
- **Predictable performance**: Consistent latency (no first-capture overhead)
- **Better resource planning**: Known memory and compute requirements
- **Easier debugging**: Graph structure is fixed and inspectable

#### Historical Context

Before CUDA graphs (CUDA 10.0, 2018), LLM serving faced a fundamental bottleneck:
- **Small batch inference was slow**: CPU overhead dominated
- **Throughput was limited**: CPU couldn't keep GPU busy
- **Latency was high**: Multiple sync points added up

CUDA graphs revolutionized LLM serving by:
- **Enabling efficient small-batch inference**: Now practical for interactive applications
- **Unlocking high-throughput serving**: CPU can handle many concurrent requests
- **Making real-time generation feasible**: Low latency enables streaming

#### The Trade-Off: When to Use CUDA Graphs

**✅ Perfect for:**
- **Decode phase**: Highly repetitive, predictable structure
- **Small batches**: Where CPU overhead dominates
- **Production serving**: High repetition, performance critical
- **Fixed model structure**: Same operations every time

**❌ Not ideal for:**
- **Prefill phase**: Variable sequence lengths, less repetitive
- **Large batches**: GPU computation dominates, CPU overhead is negligible
- **Dynamic control flow**: Operations that change based on input
- **One-off inference**: Not worth capture cost

**In Mini-SGLang:**
- CUDA graphs are used **only for decode phase** (batch_size ≤ max_graph_bs)
- Prefill uses normal forward pass (variable lengths, less repetitive)
- This hybrid approach maximizes benefits while minimizing costs

### Key Files
- **`python/minisgl/engine/graph.py`**: Main CUDA graph implementation
- **`python/minisgl/engine/engine.py`**: Integration with the engine
- **`python/minisgl/attention/fa3.py`**: Attention backend CUDA graph support

### How It Works

#### 1.1 Initialization (`GraphRunner.__init__`)

The `GraphRunner` class captures CUDA graphs for different batch sizes during initialization:

```python
# Location: python/minisgl/engine/graph.py:48-131

# Step 1: Determine batch sizes to capture
cuda_graph_bs = _determine_cuda_graph_bs(...)  # e.g., [1, 2, 4, 8, 16, ...]

# Step 2: Allocate output tensor for logits
self.logits = torch.empty((max_graph_bs, vocab_size), dtype=torch.float16, device=device)

# Step 3: Initialize attention backend for graph capture
attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=cuda_graph_bs)
```

#### 1.2 Graph Capture Process

The capture happens in a loop for each batch size:

```python
# Location: python/minisgl/engine/graph.py:110-124

for bs in cuda_graph_bs:
    g = torch.cuda.CUDAGraph()
    
    # Prepare batch with dummy requests
    batch = Batch(reqs=[dummy_req] * bs, phase="decode")
    attn_backend.prepare_for_capture(batch)  # Set up attention metadata
    
    # Capture the forward pass
    with get_global_ctx().forward_batch(batch):
        self.logits[:bs] = model.forward()  # Warm-up run
        # CUDA OPTIMIZATION HAPPENS HERE:
        # Inside this context, CUDA records all operations and optimizes them
        with torch.cuda.graph(g, pool=pool, stream=stream):
            self.logits[:bs] = model.forward()  # Actual capture
        # When exiting context, CUDA:
        # 1. Records all GPU operations that occurred
        # 2. Optimizes kernel scheduling, memory access, fusion opportunities
        # 3. Stores optimized execution plan inside 'g' object
    
    # MEMORY POOL MECHANISM:
    # If pool is None (first graph), create a new memory pool from this graph
    # If pool exists (subsequent graphs), reuse the existing pool
    if pool is None:
        pool = g.pool()  # Extract memory pool from first graph (largest batch size)
        # This pool is sized for the largest graph and can accommodate smaller graphs
    
    # STORAGE: Save the optimized graph (contains execution plan inside)
    graph_list.append((bs, g))  # 'g' now contains the optimized execution plan
```

**Key Points:**
- **Memory Pool**: All graphs share a memory pool (`pool`) to reuse memory efficiently
- **Dummy Requests**: Uses dummy requests with fixed shapes for capture
- **Decode Phase Only**: CUDA graphs are only used for decode (not prefill)

#### Understanding `torch.cuda.CUDAGraph()` and `torch.cuda.graph()`

These are the two key PyTorch APIs for CUDA graph functionality:

**1. `torch.cuda.CUDAGraph()` - The Graph Object**

```python
# Location: python/minisgl/engine/graph.py:138
g = torch.cuda.CUDAGraph()
```

**What it is:**
- A Python object that represents a CUDA graph
- Created **before** capture to hold the graph structure
- Initially empty - gets populated during capture

**What it stores (after capture):**
- **Optimized execution plan**: Sequence of GPU operations (kernels, memory transfers)
- **Kernel launch parameters**: Pre-computed grid/block sizes, shared memory configs
- **Memory addresses**: Fixed pointers to input/output tensors
- **Synchronization points**: Pre-determined sync locations
- **Memory pool reference**: Link to shared memory pool (if used)

**Key methods:**
- `g.pool()`: Returns the memory pool associated with this graph (used for sharing)
- `g.replay()`: Executes the captured graph (fast replay)

**2. `torch.cuda.graph()` - The Capture Context Manager**

```python
# Location: python/minisgl/engine/graph.py:151
with torch.cuda.graph(g, pool=pool, stream=stream):
    self.logits[:bs] = model.forward()
```

**What it does:**
- **Context manager** that captures GPU operations
- Records all operations executed inside the context
- Optimizes the recorded operations
- Stores the optimized plan in the `g` object

**Parameters:**
- `g`: The `CUDAGraph` object to populate
- `pool`: Optional memory pool for sharing memory across graphs
- `stream`: CUDA stream to execute operations on

**Inside the context manager:**
1. **Recording Phase**: All GPU operations are recorded (not executed immediately)
   - Kernel launches (matrix ops, attention, activations)
   - Memory transfers (H2D, D2H, D2D)
   - Synchronization points
2. **Optimization Phase** (when exiting context):
   - CUDA runtime analyzes the recorded operations
   - Optimizes kernel scheduling and ordering
   - Identifies memory access patterns
   - Finds fusion opportunities
   - Creates efficient execution plan
3. **Storage Phase**: Optimized plan stored in `g` object

**Memory Pool Mechanism:**

```python
# Location: python/minisgl/engine/graph.py:133, 159-160

pool = None  # Start with no pool
for bs in cuda_graph_bs:  # Iterate largest to smallest
    g = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(g, pool=pool, stream=stream):
        # Capture operations...
    
    if pool is None:
        # FIRST GRAPH (largest batch size):
        # - Creates a new memory pool sized for this graph
        # - Pool is stored inside 'g' object
        pool = g.pool()  # Extract the pool for reuse
    # SUBSEQUENT GRAPHS (smaller batch sizes):
    # - Reuse the existing 'pool' from first graph
    # - All graphs share the same memory pool
```

**Why this works:**
- **First graph** (`pool=None`): CUDA creates a memory pool sized for the largest graph
- **Subsequent graphs** (`pool=existing_pool`): Reuse the same pool (smaller graphs fit inside)
- **Result**: All graphs share one large memory pool, avoiding fragmentation

**Deep Dive: How CUDA Optimization Happens**

The optimization happens **inside CUDA runtime/driver**, not in PyTorch code. When you exit the `torch.cuda.graph()` context:

1. **Graph Construction**:
   - CUDA builds a DAG (Directed Acyclic Graph) of operations
   - Nodes = operations (kernels, memcpy)
   - Edges = dependencies (data flow, synchronization)

2. **Static Analysis**:
   - Analyzes memory access patterns
   - Identifies read-after-write, write-after-read dependencies
   - Determines which operations can run in parallel

3. **Optimization Passes**:
   - **Kernel Scheduling**: Reorders kernels for better GPU utilization
   - **Memory Coalescing**: Groups memory accesses for efficiency
   - **Kernel Fusion**: Merges adjacent operations when possible
   - **Register Allocation**: Optimizes register usage across kernels
   - **Pipeline Optimization**: Overlaps computation and memory transfers

4. **Code Generation**:
   - Generates optimized execution plan
   - Pre-computes launch parameters
   - Creates efficient kernel launch sequence

5. **Storage**:
   - Stores plan in GPU memory (inside `g` object)
   - Plan is a compact representation of the optimized execution

**Note**: The actual optimization algorithms are proprietary CUDA runtime implementations. PyTorch provides the interface (`torch.cuda.graph()`), but CUDA does the heavy lifting.

#### 1.3 Why Warm-Up? (Capture and Destroy Before Real Capture)

Before capturing the graphs we actually want to keep, there's a warm-up step:

```python
# Location: python/minisgl/engine/graph.py:109-117

# Warm up by capturing a graph and then destroying it
g = torch.cuda.CUDAGraph()
batch = Batch(reqs=[dummy_req] * self.max_graph_bs, phase="decode")
attn_backend.prepare_for_capture(batch)
with get_global_ctx().forward_batch(batch):
    self.logits[:] = model.forward()
    with torch.cuda.graph(g, stream=stream):
        self.logits[:] = model.forward()
del g  # Destroy the warm-up graph
```

**Why Do This?**

CUDA graph capture involves several initialization steps that happen lazily on the first capture:

1. **CUDA Runtime Initialization**: The first time you call `torch.cuda.graph()`, CUDA may need to:
   - Initialize internal data structures
   - Set up graph capture infrastructure
   - Allocate internal buffers for tracking operations

2. **Kernel Compilation**: Some CUDA kernels might be JIT-compiled on first use:
   - FlashAttention kernels
   - Custom CUDA operations
   - Memory allocation routines

3. **Memory Allocator Warm-Up**: CUDA's memory allocator may:
   - Initialize allocation pools
   - Establish memory fragmentation patterns
   - Cache allocation strategies

4. **PyTorch Internal State**: PyTorch's CUDA graph support may:
   - Initialize operation tracking
   - Set up memory mapping for captured operations
   - Prepare synchronization primitives

**Benefits of Warm-Up:**

✅ **Predictable Performance**: The actual capture loop has consistent timing (no first-capture overhead)

✅ **Consistent Memory State**: Memory allocator is in a "steady state" before real captures

✅ **Accurate Memory Measurements**: After warm-up, memory stats reflect the true state (see `reset_peak_memory_stats` on line 103)

✅ **Faster Capture Loop**: All lazy initialization is done, so the real captures are faster

✅ **Better Error Handling**: If there are any initialization issues, they surface during warm-up rather than during the real capture loop

**What Happens:**
1. **Warm-up capture** (lines 109-116): Captures a graph with maximum batch size, then immediately deletes it
2. **Memory cleanup** (line 117): `del g` frees the warm-up graph, but leaves CUDA runtime initialized
3. **Real captures** (lines 127-139): All subsequent captures benefit from the warm-up initialization

**Note**: The warm-up uses `max_graph_bs` (largest batch size) to ensure all initialization paths are exercised, since the largest graph will need the most resources.

**TODO/Question: Is Warm-Up a Good Trade-Off?**

While warm-up provides clear benefits, it's worth considering:

1. **Resource Cost**: The warm-up captures and immediately destroys a full graph with `max_graph_bs`. This consumes:
   - GPU memory (temporarily, until `del g`)
   - GPU compute cycles (forward pass + capture overhead)
   - Initialization time

2. **Trade-Off Analysis**: 
   - **Cost**: One extra graph capture (largest batch size) + deletion
   - **Benefit**: Predictable, faster subsequent captures + accurate memory stats + early error detection
   - **Question**: Does the one-time cost justify the benefits? Is there a measurable performance difference with vs. without warm-up?

3. **Practical Safeguard**:
   - **Early Failure Detection**: If initialization fails, it fails during warm-up (before real captures), making debugging easier
   - **Memory State Consistency**: Ensures memory allocator is in a known state before measuring/allocating for real graphs
   - **Production Reliability**: In production, one-time initialization cost is negligible compared to the reliability benefits

4. **Alternative Approaches**:
   - Could we skip warm-up and accept first-capture overhead?
   - Could we use a smaller batch size for warm-up (faster but less thorough)?
   - Are there benchmarks showing warm-up impact on total initialization time?

**Investigation Needed**: Measure and document the actual cost/benefit of warm-up in practice. Is it a necessary safeguard or an optimization that could be optional?

#### 1.4 Why Sort Batch Sizes in Descending Order? (Memory Pool Efficiency)

This is a critical optimization! Let's understand why:

```python
# Location: python/minisgl/engine/graph.py:79-82

# Sort in reverse (largest first) for memory pool efficiency
cuda_graph_bs = sorted(set(cuda_graph_bs), reverse=True)
```

**The Problem:**
When capturing CUDA graphs, each graph needs GPU memory for its intermediate tensors (activations, attention outputs, etc.). CUDA provides a **memory pool** mechanism where multiple graphs can share the same pool of GPU memory.

**How Memory Pools Work:**
1. **First graph creates the pool**: When you capture the first CUDA graph with `torch.cuda.graph(g, pool=pool)`, if `pool=None`, CUDA creates a new memory pool sized for that graph's memory needs.
2. **Subsequent graphs reuse the pool**: When you capture additional graphs with the same `pool`, they try to allocate memory from the existing pool.

**Why Descending Order Matters:**

**❌ Bad (Ascending Order - Small to Large):**
```
Capture batch_size=1  → Pool created with size for bs=1  (small pool)
Capture batch_size=2  → Tries to reuse pool, but needs more memory
Capture batch_size=4  → Pool must expand or fail
Capture batch_size=8  → Pool must expand again
Capture batch_size=16 → Pool must expand again
```
**Problems:**
- Pool must be resized multiple times (expensive!)
- Each resize requires allocating new memory and potentially copying data
- Risk of memory fragmentation
- Slower capture process

**✅ Good (Descending Order - Large to Small):**
```
Capture batch_size=16 → Pool created with size for bs=16 (large pool)
Capture batch_size=8  → Reuses pool, only needs subset of memory ✓
Capture batch_size=4  → Reuses pool, only needs subset of memory ✓
Capture batch_size=2  → Reuses pool, only needs subset of memory ✓
Capture batch_size=1  → Reuses pool, only needs subset of memory ✓
```
**Benefits:**
- Pool is created once at maximum size
- All subsequent graphs fit within the pool (no resizing needed)
- No memory fragmentation
- Faster capture process
- More predictable memory usage

**Code Flow:**
```python
# Location: python/minisgl/engine/graph.py:126-139

pool = None  # Start with no pool
for bs in cuda_graph_bs:  # Iterate largest to smallest
    g = torch.cuda.CUDAGraph()
    # ... prepare batch ...
    with torch.cuda.graph(g, pool=pool, stream=stream):
        # Capture graph operations
        self.logits[:bs] = model.forward()
    
    if pool is None:
        # First graph (largest) creates the pool
        pool = g.pool()  # Extract the memory pool
    # Subsequent graphs (smaller) reuse the same pool
```

**Real-World Impact:**
- **Memory Efficiency**: One large pool is more efficient than multiple resized pools
- **Performance**: Avoiding pool resizes saves significant time during initialization
- **Reliability**: Prevents potential out-of-memory errors from pool expansion failures

#### 1.5 Attention Backend Integration

The attention backend prepares metadata for capture and replay:

```python
# Location: python/minisgl/attention/fa3.py:115-141

def prepare_for_capture(self, batch: Batch):
    # Set up metadata using pre-allocated capture tensors
    metadata = FA3Metadata(
        cu_seqlens_k=capture.cu_seqlens_k[:bs + 1],
        cu_seqlens_q=capture.cu_seqlens_q[:bs + 1],
        positions=capture.positions[:bs],
        ...
    )
    batch.attn_metadata = metadata

def prepare_for_replay(self, batch: Batch):
    # Copy actual batch data into capture tensors before replay
    self.capture.input_ids[:bs].copy_(batch.input_ids)
    self.capture.out_loc[:bs].copy_(batch.out_loc)
    # ... copy other metadata
```

#### 1.6 Replay During Inference

During actual inference, the engine checks if CUDA graph can be used:

```python
# Location: python/minisgl/engine/engine.py:196-202

def forward_batch(self, batch: Batch, args: BatchSamplingArgs):
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            # Use CUDA graph replay (fast!)
            logits = self.graph_runner.replay(batch)
        else:
            # Fall back to normal forward pass
            logits = self.model.forward()
```

The replay function:

```python
# Location: python/minisgl/engine/graph.py:147-152

def replay(self, batch: Batch) -> torch.Tensor:
    g = self.graph_map[batch.padded_size]  # Get graph for this batch size
    self.attn_backend.prepare_for_replay(batch)  # Update capture tensors
    g.replay()  # Replay the captured graph
    return self.logits[:batch.size]
```

#### Why Is CUDA Graph Replay Fast?

CUDA graph replay (`g.replay()`) is significantly faster than a normal forward pass for several reasons:

**1. Eliminates CPU Launch Overhead**

**Normal Forward Pass:**
```
CPU: Launch kernel 1 → Wait → Launch kernel 2 → Wait → Launch kernel 3 → ...
     [CPU-GPU sync]      [CPU-GPU sync]      [CPU-GPU sync]
```
- Each kernel launch requires CPU-GPU communication
- CPU must wait for GPU to be ready before launching next kernel
- High latency for small batch sizes (CPU overhead dominates)

**CUDA Graph Replay:**
```
CPU: Launch entire graph once → Done
     [Single CPU-GPU sync]
GPU: Execute all kernels in sequence (pre-scheduled)
```
- Single launch command for the entire computation graph
- GPU executes all kernels back-to-back without CPU intervention
- Minimal CPU involvement after the initial launch

**2. Pre-Compiled Execution Plan**

**Where Optimization Happens:**
```python
# Location: python/minisgl/engine/graph.py:144-145
with torch.cuda.graph(g, pool=pool, stream=stream):
    self.logits[:bs] = model.forward()  # Operations recorded here
# CUDA optimizes when exiting this context manager
```

**During capture (inside `torch.cuda.graph()` context), CUDA:**
- **Records** the exact sequence of operations (all kernels, memory transfers)
- **Optimizes** kernel launch parameters (grid/block sizes, shared memory)
- **Pre-determines** memory access patterns (coalescing, prefetching)
- **Creates** an efficient execution schedule (kernel ordering, overlap)
- **Stores** the optimized plan inside the `g` object (in GPU memory)

**What CUDA Optimizes (Automatic, Inside CUDA Runtime):**
- **Kernel Scheduling**: Optimal execution order and overlap opportunities
- **Memory Coalescing**: Grouping memory accesses for efficiency
- **Kernel Fusion**: Identifying operations that can be merged
- **Register Allocation**: Efficient register usage across kernels
- **Pipeline Optimization**: Overlapping computation and memory transfers
- **Stream Synchronization**: Minimizing synchronization overhead

**Storage Location:**
```python
# Location: python/minisgl/engine/graph.py:153
self.graph_map = dict(graph_list)  # Maps batch_size → optimized graph object
# Each graph object 'g' contains the optimized execution plan internally
```

**During replay, this optimized plan is executed directly without:**
- Re-evaluating which kernels to launch (plan is pre-determined)
- Re-computing launch configurations (parameters are pre-optimized)
- Re-determining execution order (schedule is pre-computed)

**3. Reduced Synchronization Points**

**Normal Forward Pass:**
- Each layer may require synchronization
- CPU checks completion before launching next operation
- Multiple CPU-GPU round trips

**CUDA Graph Replay:**
- All operations are pre-scheduled
- GPU executes continuously without CPU checks
- Single synchronization point at the end

**4. Kernel Fusion Opportunities**

CUDA can optimize the entire graph as a unit:
- **Kernel Merging**: Adjacent operations can be fused into single kernels
- **Memory Coalescing**: Better memory access patterns across the graph
- **Register Optimization**: More efficient register usage across kernels
- **Pipeline Optimization**: Better overlap of computation and memory transfers

**5. Better Memory Access Patterns**

- **Prefetching**: CUDA can prefetch data for upcoming kernels
- **Locality**: Memory accesses are known in advance, enabling better caching
- **Reduced Memory Fragmentation**: Pre-allocated memory pool reduces allocation overhead
- **Cache Efficiency**: Predictable access patterns improve GPU cache hit rates

**6. Lower Latency for Small Batches**

This is especially critical for decode phase (batch size = 1):
- **Normal forward**: CPU overhead can be 50-80% of total time for batch_size=1
- **Graph replay**: CPU overhead is minimal (~5-10% of total time)
- **Result**: 2-5x speedup for small batch inference

**Performance Comparison:**

```
Normal Forward Pass (batch_size=1):
├─ CPU launch overhead: ~200-500μs
├─ GPU computation: ~500-1000μs
└─ Total: ~700-1500μs

CUDA Graph Replay (batch_size=1):
├─ CPU launch overhead: ~10-50μs
├─ GPU computation: ~500-1000μs (same)
└─ Total: ~510-1050μs

Speedup: 1.4x - 2.9x for small batches
```

**Why This Matters for LLM Serving:**

- **High Throughput**: Decode phase runs thousands of times per second
- **Low Latency**: Every microsecond saved improves user experience
- **Scalability**: Lower CPU overhead means CPU can handle more concurrent requests
- **Resource Efficiency**: More GPU utilization, less CPU waiting

**Trade-Off:**

- **Memory**: CUDA graphs consume GPU memory (but shared pool minimizes this)
- **Flexibility**: Graph structure is fixed (but decode phase is predictable)
- **Initialization**: One-time capture cost (but amortized over many replays)

For decode-phase inference (which is highly repetitive), the benefits far outweigh the costs.

**Why Batch Padding?**
- CUDA graphs are captured for specific batch sizes (e.g., 1, 2, 4, 8, ...)
- If actual batch size is 3, it's padded to 4 to use the graph for batch size 4
- See `pad_batch()` method in `graph.py:159-166`

### Benefits
- **Reduced CPU Overhead**: Eliminates repeated CPU-GPU synchronization
- **Faster Decode**: Critical for high-throughput serving
- **Memory Efficient**: Shared memory pool across all graphs (optimized by descending batch size order)
- **Single Output Tensor**: All graphs share `self.logits` tensor, avoiding per-graph allocations

---

## 2. Tensor Parallelism (Distributed Serving)

### Overview
Tensor Parallelism splits model weights and computation across multiple GPUs. Each GPU holds a shard of the model and communicates with others via NCCL (or PyNCCL) for all-reduce and all-gather operations.

### Key Files
- **`python/minisgl/distributed/impl.py`**: Communication primitives
- **`python/minisgl/distributed/info.py`**: TP rank/size information
- **`python/minisgl/layers/linear.py`**: TP-aware linear layers
- **`python/minisgl/engine/engine.py`**: TP initialization

### Architecture

#### 2.1 TP Information Setup

Each process knows its rank and world size:

```python
# Location: python/minisgl/distributed/info.py

@dataclass(frozen=True)
class DistributedInfo:
    rank: int  # 0, 1, 2, ... (TP rank)
    size: int  # Total number of TP workers

# Set during engine initialization
set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
```

#### 2.2 Communication Primitives

Two implementations are available:

**A. Torch Distributed (NCCL backend)**
```python
# Location: python/minisgl/distributed/impl.py:25-41

class TorchDistributedImpl:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x
    
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out
```

**B. PyNCCL (Custom CUDA implementation)**
```python
# Location: python/minisgl/distributed/impl.py:45-60

class PyNCCLDistributedImpl:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")  # Custom CUDA kernel
        return x
```

#### 2.3 Weight Sharding

Model weights are sharded across TP ranks:

**Column Parallel (QKV projection):**
```python
# Location: python/minisgl/layers/linear.py:50-67

class LinearQKVMerged:
    def __init__(self, hidden_size, head_dim, num_qo_heads, num_kv_heads, ...):
        tp_info = get_tp_info()
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        # Each rank gets 1/TP_SIZE of the KV heads
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
```

**Row Parallel (Output projection):**
```python
# Location: python/minisgl/layers/linear.py:88-106

class LinearRowParallel:
    def __init__(self, input_size, output_size, ...):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        # Each rank gets 1/TP_SIZE of input dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)  # Sum across TP ranks
        return y
```

**Output Parallel (LM Head):**
```python
# Location: python/minisgl/layers/linear.py:70-85

class LinearOProj:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)  # Sum partial outputs
        return y
```

#### 2.4 Attention with TP

Attention layers split heads across TP ranks:

```python
# Location: python/minisgl/layers/attention.py:32-34

tp_size = get_tp_info().size
self.num_qo_heads = divide_even(num_qo_heads, tp_size)
self.num_kv_heads = divide_even(num_kv_heads, tp_size)
```

Each TP rank computes attention for its subset of heads, then results are combined via all-reduce.

#### 2.5 Embedding with TP

Vocab embeddings are also sharded:

```python
# Location: python/minisgl/layers/embedding.py:14-41

class VocabParallelEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        tp_info = get_tp_info()
        self.num_embeddings_tp = divide_up(num_embeddings, tp_info.size)
        start_idx = self.num_embeddings_tp * tp_rank
        # Each rank holds a portion of the vocabulary
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = indexing(weights=self.weight, indices=x, vocab_range=...)
        return self._comm.all_reduce(y)  # Gather embeddings
```

#### 2.6 Initialization Flow

```python
# Location: python/minisgl/engine/engine.py:114-140

def _init_communication(self, config: EngineConfig):
    if config.tp_info.size == 1:
        # Single GPU, no communication needed
        return
    
    # Initialize PyTorch distributed (NCCL backend)
    torch.distributed.init_process_group(
        backend="nccl",
        rank=config.tp_info.rank,
        world_size=config.tp_info.size,
        ...
    )
    
    # Optionally enable PyNCCL for custom CUDA kernels
    if config.use_pynccl:
        enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
```

### Communication Pattern

**Forward Pass:**
1. Input embeddings: All-gather (each rank needs full vocab)
2. QKV projection: Column parallel (no communication)
3. Attention: No communication (heads are independent)
4. Output projection: All-reduce (sum partial outputs)
5. LM head: All-reduce (sum logits)

**Key Insight**: Communication happens at layer boundaries, not within attention computation.

---

## 3. Radix Cache (Prefix Caching)

### Overview
Radix Cache enables KV cache reuse for shared prefixes across requests. It uses a radix tree (trie) data structure to efficiently store and match token sequences.

### Key Files
- **`python/minisgl/kvcache/radix_manager.py`**: Radix tree implementation
- **`python/minisgl/scheduler/cache.py`**: Cache manager integration
- **`python/minisgl/kvcache/base.py`**: Base interfaces

### Data Structure: Radix Tree

#### 3.1 RadixTreeNode

Each node represents a contiguous sequence of tokens:

```python
# Location: python/minisgl/kvcache/radix_manager.py:13-76

class RadixTreeNode:
    def __init__(self):
        self.children: Dict[int, RadixTreeNode] = {}  # token_id -> child node
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0  # Number of active requests using this node
        self.timestamp: int  # For LRU eviction
        self._key: torch.Tensor  # Token sequence this node represents
        self._value: torch.Tensor  # KV cache page indices
        self._length: int  # Length of the sequence
```

**Key Properties:**
- **ref_count**: Tracks how many requests are using this prefix (prevents eviction)
- **timestamp**: Used for LRU eviction when cache is full
- **key/value**: The actual token sequence and corresponding cache indices

#### 3.2 Tree Walking (`_walk`)

The core algorithm for finding matching prefixes:

```python
# Location: python/minisgl/kvcache/radix_manager.py:138-163

def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
    prefix_len = 0
    node = self.root_node
    
    while prefix_len < len(input_ids):
        this_id = int(input_ids[prefix_len].item())
        
        # Check if child exists for this token
        if this_id not in node.children:
            return node, prefix_len  # No match, return current node
        
        node = node.children[this_id]
        
        # Compare full sequence (handles partial matches)
        match_len = node.get_match_len(input_ids[prefix_len:])
        prefix_len += match_len
        
        # Handle partial match (need to split node)
        if match_len != node.length:
            node = node._split_at(match_len)  # Split at mismatch point
            return node, prefix_len
        
        # Full match, continue walking
        node.timestamp = time.monotonic_ns()  # Update access time
    
    return node, prefix_len  # Fully matched
```

**Example:**
- Tree has: `[1, 2, 3]` → cache indices `[10, 11, 12]`
- New request: `[1, 2, 4]`
- Walk finds node `[1, 2, 3]`, but `3 != 4`
- Node is split: `[1, 2]` (parent) → `[3]` (child) and `[4]` (new child)

#### 3.3 Prefix Matching

```python
# Location: python/minisgl/kvcache/radix_manager.py:116-126

def match_prefix(self, input_ids: torch.Tensor):
    node, prefix_len = self._walk(input_ids)
    
    if prefix_len == 0:
        return RadixCacheHandle(prefix_len, node), empty_tensor
    
    # Collect cache indices by walking up the tree
    value_list = []
    while not node.is_root():
        value_list.append(node.value)  # Cache indices for this segment
        node = node.parent
    value_list.reverse()  # Restore correct order
    
    return RadixCacheHandle(prefix_len, node), torch.cat(value_list)
```

**Returns:**
- **handle**: Reference to the matched node (for locking)
- **indices**: Concatenated cache indices for the matched prefix

#### 3.4 Reference Counting (Locking)

Prevents eviction of actively used prefixes:

```python
# Location: python/minisgl/kvcache/radix_manager.py:97-114

def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False):
    node = handle.node
    
    if unlock:
        # Decrement ref_count up the tree
        while not node.is_root():
            node = node.parent
            node.ref_count -= 1
            if node.ref_count == 0:
                # Now evictable
                self.evictable_size += node.length
                self.protected_size -= node.length
    else:
        # Increment ref_count up the tree
        while not node.is_root():
            node = node.parent
            if node.ref_count == 0:
                # Was evictable, now protected
                self.evictable_size -= node.length
                self.protected_size += node.length
            node.ref_count += 1
```

**Why walk up the tree?**
- If a request uses prefix `[1, 2, 3]`, all parent nodes `[1]`, `[1, 2]`, `[1, 2, 3]` must be protected
- This ensures shared prefixes remain available

#### 3.5 Insertion

```python
# Location: python/minisgl/kvcache/radix_manager.py:128-136

def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
    node, prefix_len = self._walk(input_ids)
    
    # Only insert the new part (not already cached)
    if prefix_len < len(input_ids):
        new_node = RadixTreeNode()
        new_node.set_key_value(
            input_ids[prefix_len:],  # New tokens
            indices[prefix_len:]      # New cache indices
        )
        new_node.set_parent(node)
        self.evictable_size += new_node.length
    
    return prefix_len  # Return how much was already cached
```

#### 3.6 Eviction (LRU)

When cache is full, evict least recently used leaf nodes:

```python
# Location: python/minisgl/kvcache/radix_manager.py:165-192

def evict(self, size: int) -> torch.Tensor:
    # Collect all evictable leaf nodes
    leave_nodes = self._collect_leave_nodes_for_evict()
    heapq.heapify(leave_nodes)  # Min-heap by timestamp (LRU)
    
    evicted_indices = []
    evicted_size = 0
    
    while evicted_size < size:
        node = heapq.heappop(leave_nodes)  # Get oldest leaf
        assert node.ref_count == 0  # Must be unused
        
        evicted_size += node.length
        evicted_indices.append(node.value)
        self.evictable_size -= node.length
        
        # Remove from parent
        parent = node.parent
        del parent.children[int(node._key[0].item())]
        
        # If parent becomes evictable leaf, add to heap
        if parent.is_leaf() and parent.ref_count == 0:
            heapq.heappush(leave_nodes, parent)
    
    return torch.cat(evicted_indices)
```

**Eviction Strategy:**
- Only evict leaf nodes (complete sequences)
- Use LRU (Least Recently Used) based on timestamp
- After eviction, check if parent becomes evictable

#### 3.7 Integration with Scheduler

The scheduler uses the cache manager:

```python
# Location: python/minisgl/scheduler/cache.py:24-62

class CacheManager:
    def match_req(self, req: PendingReq):
        # Find matching prefix in radix tree
        handle, indices = self.manager.match_prefix(req.input_ids[:input_len - 1])
        return handle, indices
    
    def lock(self, handle: BaseCacheHandle):
        # Protect prefix from eviction
        self.manager.lock_handle(handle, unlock=False)
    
    def allocate(self, needed_len: int) -> torch.Tensor:
        # Try free slots first
        if needed_len <= len(self._free_slots):
            return self._free_slots[:needed_len]
        
        # Evict if needed
        evicted = self.manager.evict(needed_len - len(self._free_slots))
        return torch.cat([self._free_slots, evicted])[:needed_len]
    
    def free_and_cache_finished_req(self, old_handle, input_ids, indices):
        # Insert finished request into radix tree
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        # Free pages that were already cached (duplicate)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)
```

### Benefits

1. **Memory Efficiency**: Shared prefixes stored once, not duplicated per request
2. **Computation Savings**: Skip recomputation for cached prefixes
3. **Dynamic Sharing**: Automatically detects and shares common prefixes
4. **LRU Eviction**: Efficiently manages cache when full

### Example Scenario

**Request 1**: "What is the capital of France?"
- Tokens: `[101, 202, 303, 404, 505]`
- Cache: `[10, 11, 12, 13, 14]`

**Request 2**: "What is the capital of Germany?"
- Tokens: `[101, 202, 303, 404, 606]` (shared prefix: `[101, 202, 303, 404]`)
- Match: Prefix `[101, 202, 303, 404]` → reuse cache `[10, 11, 12, 13]`
- Only compute: `[606]` → cache `[15]`

**Radix Tree Structure:**
```
root
 └─ [101]
     └─ [202]
         └─ [303]
             └─ [404]
                 ├─ [505] → cache [14] (Request 1)
                 └─ [606] → cache [15] (Request 2)
```

---

## Summary

1. **CUDA Graph**: Pre-captures decode-phase operations to eliminate CPU overhead. Uses descending batch size order for optimal memory pool efficiency.
2. **Tensor Parallelism**: Shards model across GPUs with communication at layer boundaries.
3. **Radix Cache**: Uses a radix tree to share KV cache for common prefixes across requests.

All three features work together to enable efficient, high-throughput LLM serving.

