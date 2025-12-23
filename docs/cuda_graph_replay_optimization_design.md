# CUDA Graph Replay Optimization Design

## Current Implementation Analysis

### Current Replay Flow

```
engine.forward_batch()
  ├─ Check: can_use_cuda_graph(batch)?
  ├─ If yes:
  │   ├─ graph_runner.replay(batch)
  │   │   ├─ Get graph: graph_map[batch.padded_size]
  │   │   ├─ attn_backend.prepare_for_replay(batch)  # ⚠️ Multiple copy operations
  │   │   │   ├─ capture.input_ids[:bs].copy_(batch.input_ids)
  │   │   │   ├─ capture.out_loc[:bs].copy_(batch.out_loc)
  │   │   │   ├─ capture.cu_seqlens_k[:bs+1].copy_(metadata.cu_seqlens_k)
  │   │   │   ├─ capture.positions[:bs].copy_(metadata.positions)
  │   │   │   ├─ capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
  │   │   │   └─ capture.page_table[:bs, :].copy_(metadata.page_table)
  │   │   ├─ g.replay()  # ✅ Fast CUDA graph replay
  │   │   └─ return logits[:batch.size]
  │   └─ Sample and return
```

### Identified Bottlenecks

#### 1. **Synchronous Memory Copies in `prepare_for_replay()`** ⚠️ HIGH IMPACT
**Current Issue:**
- 6-7 synchronous `.copy_()` operations before each replay
- Each copy is a separate GPU kernel launch
- Blocks until all copies complete before replay starts
- Estimated overhead: ~10-50μs per replay

**Location:** `python/minisgl/attention/fa3.py:136-147`

**Impact:** For decode phase running thousands of times/second, this adds significant latency.

#### 2. **Context Manager Overhead** ⚠️ MEDIUM IMPACT
**Current Issue:**
- `with self.ctx.forward_batch(batch):` context manager
- Python context manager overhead (enter/exit)
- Property access for `get_global_ctx().batch`

**Location:** `python/minisgl/engine/engine.py:200`

**Impact:** Small but measurable overhead on hot path.

#### 3. **Metadata Preparation Timing** ⚠️ MEDIUM IMPACT
**Current Issue:**
- `prepare_for_replay()` happens synchronously before `g.replay()`
- Could potentially overlap with previous computation
- No pipelining of data preparation

**Impact:** Missed opportunity for overlap.

#### 4. **Tensor Slicing** ⚠️ LOW IMPACT
**Current Issue:**
- `self.logits[:batch.size]` creates a view
- View creation has minimal overhead but still present

**Location:** `python/minisgl/engine/graph.py:190`

**Impact:** Negligible, but could be optimized.

#### 5. **Graph Lookup** ✅ LOW IMPACT (Already Optimized)
**Current Issue:**
- Dictionary lookup `graph_map[batch.padded_size]`
- Already fast (O(1) hash lookup)

**Impact:** Minimal, acceptable.

## Proposed Optimizations

### Optimization 1: Async Memory Copies with Stream Overlap ⭐ HIGH PRIORITY

**Design:**
- Use CUDA streams to overlap memory copies with previous computation
- Batch multiple copies into a single operation where possible
- Use `non_blocking=True` for copies that don't need immediate synchronization

**Implementation Approach:**
```python
def prepare_for_replay_async(self, batch: Batch, stream: torch.cuda.Stream) -> None:
    """Async version that overlaps copies with previous work"""
    metadata, bs = batch.attn_metadata, batch.padded_size
    
    with torch.cuda.stream(stream):
        # Batch small copies together
        # Use non_blocking where possible
        self.capture.input_ids[:bs].copy_(batch.input_ids, non_blocking=True)
        self.capture.out_loc[:bs].copy_(batch.out_loc, non_blocking=True)
        # ... other copies
```

**Benefits:**
- Overlap copies with previous GPU computation
- Reduce perceived latency
- Better GPU utilization

**Challenges:**
- Need to ensure synchronization before replay
- More complex stream management
- Need to track which copies are async

**Estimated Speedup:** 10-30% reduction in replay latency

---

### Optimization 2: Fused Copy Operations ⭐ HIGH PRIORITY

**Design:**
- Combine multiple small copies into fewer, larger copy operations
- Use custom CUDA kernel to copy multiple tensors in one launch
- Reduce kernel launch overhead

**Implementation Approach:**
```python
def prepare_for_replay_fused(self, batch: Batch) -> None:
    """Fused copy operation using custom kernel"""
    # Single kernel launch that copies all data
    # Custom CUDA kernel: copy_multiple_tensors()
    copy_multiple_tensors(
        src=[batch.input_ids, batch.out_loc, metadata.cu_seqlens_k, ...],
        dst=[capture.input_ids, capture.out_loc, capture.cu_seqlens_k, ...],
        sizes=[bs, bs, bs+1, ...]
    )
```

**Benefits:**
- Single kernel launch instead of 6-7 launches
- Reduced CPU overhead
- Better memory bandwidth utilization

**Challenges:**
- Need to write custom CUDA kernel
- More complex implementation
- Need to handle variable sizes

**Estimated Speedup:** 20-40% reduction in copy overhead

---

### Optimization 3: Pre-compute Constant Metadata ⭐ MEDIUM PRIORITY

**Design:**
- Identify metadata that doesn't change between replays
- Pre-compute and cache constant values
- Only copy what actually changes

**Analysis:**
- `cu_seqlens_q`: Always `[0, 1, 2, ..., bs]` for decode - **CONSTANT**
- `positions`: Changes each step - **VARIABLE**
- `cache_seqlens`: Changes as sequences grow - **VARIABLE**
- `page_table`: May change - **VARIABLE**

**Implementation Approach:**
```python
def prepare_for_replay(self, batch: Batch) -> None:
    # Skip cu_seqlens_q copy (it's always the same for decode)
    # Only copy what actually changes
    self.capture.positions[:bs].copy_(metadata.positions)
    self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
    # ... only variable data
```

**Benefits:**
- Fewer copy operations
- Less memory bandwidth used
- Simpler code path

**Challenges:**
- Need to carefully identify what's constant
- May need per-backend analysis

**Estimated Speedup:** 10-15% reduction in copy overhead

---

### Optimization 4: Eliminate Context Manager for Replay Path ⭐ MEDIUM PRIORITY

**Design:**
- For CUDA graph replay, we know the batch is already set
- Skip context manager overhead
- Directly set batch if needed

**Implementation Approach:**
```python
def replay(self, batch: Batch) -> torch.Tensor:
    # Fast path: assume batch is already set in context
    # Or set it directly without context manager
    g = self.graph_map[batch.padded_size]
    self.attn_backend.prepare_for_replay(batch)
    g.replay()
    return self.logits[:batch.size]
```

**Benefits:**
- Eliminate Python context manager overhead
- Simpler code path
- Slightly faster

**Challenges:**
- Need to ensure batch is set correctly
- May need to refactor context management

**Estimated Speedup:** 1-3% overall improvement

---

### Optimization 5: Pipeline Data Preparation ⭐ MEDIUM PRIORITY

**Design:**
- Start preparing data for next replay while current replay is executing
- Use double buffering for capture tensors
- Overlap preparation with computation

**Implementation Approach:**
```python
class GraphRunner:
    def __init__(self, ...):
        # Double buffer for capture data
        self.capture_buffers = [capture_data_1, capture_data_2]
        self.current_buffer = 0
    
    def prepare_for_replay_async(self, batch: Batch, next_batch: Batch):
        # Prepare next_batch while current replay runs
        # Switch buffers between replays
```

**Benefits:**
- Hide data preparation latency
- Better GPU utilization
- Lower end-to-end latency

**Challenges:**
- More complex state management
- Need to handle batch ordering
- May not always have next batch ready

**Estimated Speedup:** 15-25% reduction in perceived latency

---

### Optimization 6: Batch Metadata Updates ⭐ LOW PRIORITY

**Design:**
- Instead of individual tensor copies, batch all metadata updates
- Use a single operation to update all capture tensors
- Reduce function call overhead

**Implementation Approach:**
```python
def prepare_for_replay_batched(self, batch: Batch) -> None:
    # Single function that does all updates
    # Potentially use torch.cat or similar to batch operations
    update_all_capture_tensors(
        capture=self.capture,
        batch=batch,
        metadata=batch.attn_metadata
    )
```

**Benefits:**
- Cleaner API
- Potentially better compiler optimization
- Less function call overhead

**Challenges:**
- May not provide significant speedup
- More complex to implement

**Estimated Speedup:** 2-5% improvement

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Low Risk, High Impact)
1. **Optimization 3**: Pre-compute constant metadata
   - Easy to implement
   - Low risk
   - Immediate benefit

2. **Optimization 4**: Eliminate context manager overhead
   - Simple change
   - Low risk
   - Small but measurable benefit

### Phase 2: High Impact Optimizations (Medium Risk)
3. **Optimization 1**: Async memory copies
   - Medium complexity
   - Requires stream management
   - Significant benefit

4. **Optimization 2**: Fused copy operations
   - High complexity (custom CUDA kernel)
   - Requires testing
   - Very significant benefit

### Phase 3: Advanced Optimizations (Higher Risk)
5. **Optimization 5**: Pipeline data preparation
   - High complexity
   - Requires scheduler changes
   - Significant benefit but complex

6. **Optimization 6**: Batch metadata updates
   - Low priority
   - May not be worth the effort

## Metrics to Track

1. **Replay Latency**: Time from `replay()` call to completion
2. **Copy Overhead**: Time spent in `prepare_for_replay()`
3. **GPU Utilization**: Percentage of time GPU is busy
4. **End-to-End Latency**: Total time for decode step
5. **Throughput**: Tokens/second processed

## Testing Strategy

1. **Unit Tests**: Test each optimization in isolation
2. **Integration Tests**: Test with real workloads
3. **Benchmarking**: Compare before/after metrics
4. **Correctness Tests**: Ensure numerical correctness maintained
5. **Stress Tests**: Test with various batch sizes and configurations

## Questions to Answer Before Implementation

1. **Are all copies necessary?** Can we eliminate any?
2. **What's the actual overhead?** Profile to measure real impact
3. **Stream management**: How to handle multiple streams safely?
4. **Backward compatibility**: Can we maintain API compatibility?
5. **Error handling**: How to handle failures in async operations?

## Next Steps

1. **Profile current implementation** to get baseline metrics
2. **Implement Optimization 3** (quick win)
3. **Measure impact** of each optimization
4. **Iterate** based on results

