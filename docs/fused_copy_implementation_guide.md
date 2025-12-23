# Fused Copy Operations Implementation Guide

## Overview
Combine 6-7 individual `.copy_()` operations into a single CUDA kernel launch to reduce overhead.

## Current State Analysis

### What We're Copying (from `fa3.py:136-141`)
```python
self.capture.input_ids[:bs].copy_(batch.input_ids)           # int32, size=bs
self.capture.out_loc[:bs].copy_(batch.out_loc)               # int32, size=bs  
self.capture.cu_seqlens_k[:bs+1].copy_(metadata.cu_seqlens_k) # int32, size=bs+1
self.capture.positions[:bs].copy_(metadata.positions)          # int32, size=bs
self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)      # int32, size=bs
self.capture.page_table[:bs, :max_seqlen_k].copy_(metadata.page_table) # int32, 2D
```

**Observations:**
- All are `int32` tensors
- Most are 1D (except `page_table` which is 2D)
- Sizes vary but are bounded by `bs` and `max_seqlen_k`
- All are contiguous memory copies

## Step-by-Step Implementation Plan

### Phase 1: Design & Planning ✅

**Step 1.1: Analyze Copy Patterns**
- [x] Identify all copy operations
- [x] Determine data types (all int32)
- [x] Determine sizes (bs, bs+1, 2D)
- [ ] Measure current overhead (benchmark baseline)

**Step 1.2: Design Kernel Interface**
- [ ] Decide on kernel signature
- [ ] Design parameter structure
- [ ] Plan for variable sizes

**Action Items:**
1. Add a simple benchmark to measure current `prepare_for_replay()` time
2. Document the exact tensor shapes and sizes

---

### Phase 2: Create Kernel Infrastructure

**Step 2.1: Create CUDA Kernel File**

**File:** `python/minisgl/kernel/csrc/jit/fused_copy.cu`

**Design Decisions:**
- **Kernel Type**: Single kernel that handles all copies
- **Threading**: One thread per element (or warp-based for efficiency)
- **Template Parameters**: 
  - Element size (4 bytes for int32)
  - Number of copy operations (6)
  - Max batch size (for bounds checking)

**Kernel Structure:**
```cuda
// Pseudo-code structure
template <...>
__global__ void fused_copy_kernel(
    // Source pointers
    const int32_t* src_input_ids,
    const int32_t* src_out_loc,
    const int32_t* src_cu_seqlens_k,
    const int32_t* src_positions,
    const int32_t* src_seq_lens,
    const int32_t* src_page_table,
    
    // Destination pointers
    int32_t* dst_input_ids,
    int32_t* dst_out_loc,
    int32_t* dst_cu_seqlens_k,
    int32_t* dst_positions,
    int32_t* dst_seq_lens,
    int32_t* dst_page_table,
    
    // Sizes
    int batch_size,
    int max_seqlen_k
) {
    // Each thread handles one element from one tensor
    // Or use warps for better efficiency
}
```

**Action Items:**
1. Create `fused_copy.cu` file
2. Start with a simple version (one copy operation)
3. Test it works before adding more

---

**Step 2.2: Create Python Wrapper**

**File:** `python/minisgl/kernel/fused_copy.py`

**Pattern to Follow:** Similar to `store.py`

```python
@lru_cache(maxsize=None)
def _jit_fused_copy_module(...) -> Module:
    # JIT compile the kernel
    pass

def fused_copy_replay_data(
    # Source tensors
    src_input_ids: torch.Tensor,
    src_out_loc: torch.Tensor,
    src_cu_seqlens_k: torch.Tensor,
    src_positions: torch.Tensor,
    src_seq_lens: torch.Tensor,
    src_page_table: torch.Tensor,
    
    # Destination tensors
    dst_input_ids: torch.Tensor,
    dst_out_loc: torch.Tensor,
    dst_cu_seqlens_k: torch.Tensor,
    dst_positions: torch.Tensor,
    dst_seq_lens: torch.Tensor,
    dst_page_table: torch.Tensor,
    
    batch_size: int,
    max_seqlen_k: int,
) -> None:
    # Launch the fused kernel
    pass
```

**Action Items:**
1. Create `fused_copy.py` following the pattern from `store.py`
2. Use `load_jit()` with `fused_copy.cu`
3. Add to `kernel/__init__.py` exports

---

### Phase 3: Incremental Development & Testing

**Step 3.1: Start with Single Copy (Proof of Concept)**

**Goal:** Get one copy working first

1. **Modify kernel** to only copy `input_ids`
2. **Test** that it works correctly
3. **Benchmark** vs. single `.copy_()` call
4. **Verify** correctness with unit test

**Test Case:**
```python
def test_fused_copy_single():
    bs = 4
    src = torch.randint(0, 1000, (bs,), dtype=torch.int32, device="cuda")
    dst = torch.zeros(bs, dtype=torch.int32, device="cuda")
    
    fused_copy_replay_data(
        src_input_ids=src,
        dst_input_ids=dst,
        # ... other params with dummy tensors
    )
    
    assert torch.equal(src, dst)
```

**Action Items:**
1. Implement single copy in kernel
2. Write unit test
3. Verify it works
4. Measure performance

---

**Step 3.2: Add Second Copy**

**Goal:** Add `out_loc` copy

1. **Extend kernel** to handle two copies
2. **Update tests** to verify both
3. **Compare** performance vs. two `.copy_()` calls

**Action Items:**
1. Add second copy to kernel
2. Update test
3. Verify correctness and performance

---

**Step 3.3: Add Remaining 1D Copies**

**Goal:** Add `cu_seqlens_k`, `positions`, `seq_lens`

**Challenge:** `cu_seqlens_k` has size `bs+1` (different from others)

**Design Decision:**
- Handle variable sizes in kernel
- Use separate loops or conditional logic
- Or use offset-based indexing

**Action Items:**
1. Add remaining 1D copies
2. Handle `bs+1` size for `cu_seqlens_k`
3. Test all 1D copies together

---

**Step 3.4: Add 2D Copy (page_table)**

**Goal:** Handle 2D tensor copy

**Challenge:** `page_table` is 2D: `[bs, max_seqlen_k]`

**Design Options:**
1. **Flatten approach**: Treat as 1D with size `bs * max_seqlen_k`
2. **Separate kernel**: Use different kernel for 2D
3. **Hybrid**: Handle 2D in same kernel with different indexing

**Recommendation:** Start with flatten approach (simplest)

**Action Items:**
1. Add 2D copy handling
2. Test with various `max_seqlen_k` values
3. Verify correctness

---

### Phase 4: Integration

**Step 4.1: Integrate into FA3 Backend**

**File:** `python/minisgl/attention/fa3.py`

**Modify `prepare_for_replay()`:**
```python
def prepare_for_replay(self, batch: Batch) -> None:
    # Option 1: Always use fused copy
    from minisgl.kernel import fused_copy_replay_data
    fused_copy_replay_data(...)
    
    # Option 2: Feature flag (for testing)
    if USE_FUSED_COPY:
        fused_copy_replay_data(...)
    else:
        # Original code
        self.capture.input_ids[:bs].copy_(batch.input_ids)
        # ...
```

**Action Items:**
1. Add import for fused copy function
2. Replace copy operations with fused call
3. Add feature flag for easy A/B testing
4. Test integration

---

**Step 4.2: Add Feature Flag**

**Purpose:** Easy way to enable/disable for testing

**Implementation:**
```python
# In fa3.py or config
USE_FUSED_COPY = os.getenv("MINISGL_USE_FUSED_COPY", "0") == "1"
```

**Action Items:**
1. Add environment variable flag
2. Test both paths work
3. Benchmark both paths

---

### Phase 5: Optimization & Refinement

**Step 5.1: Optimize Kernel Launch Parameters**

**Consider:**
- Block size (threads per block)
- Grid size (number of blocks)
- Occupancy
- Memory coalescing

**Action Items:**
1. Profile kernel performance
2. Tune launch parameters
3. Measure improvement

---

**Step 5.2: Handle Edge Cases**

**Edge Cases to Consider:**
- `bs=0` (shouldn't happen, but be safe)
- `bs > max_graph_bs` (should be caught earlier)
- `max_seqlen_k=0` (edge case)
- Non-contiguous tensors (shouldn't happen, but verify)

**Action Items:**
1. Add bounds checking
2. Add assertions
3. Test edge cases

---

**Step 5.3: Benchmark & Compare**

**Metrics to Measure:**
1. **Latency**: Time for `prepare_for_replay()`
2. **Throughput**: Copies per second
3. **GPU Utilization**: During copy operations
4. **End-to-End**: Impact on decode latency

**Benchmark Script:**
```python
def benchmark_copy_operations():
    # Warm up
    for _ in range(10):
        prepare_for_replay(batch)
    
    # Time original
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        prepare_for_replay_original(batch)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # Time fused
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        prepare_for_replay_fused(batch)
    torch.cuda.synchronize()
    fused_time = time.time() - start
    
    print(f"Original: {original_time*1000:.2f}ms")
    print(f"Fused: {fused_time*1000:.2f}ms")
    print(f"Speedup: {original_time/fused_time:.2f}x")
```

**Action Items:**
1. Create benchmark script
2. Run on various batch sizes
3. Compare results
4. Document improvements

---

## Implementation Checklist

### Phase 1: Planning ✅
- [x] Analyze current copy operations
- [ ] Create benchmark baseline
- [ ] Document tensor shapes/sizes

### Phase 2: Infrastructure
- [ ] Create `fused_copy.cu` kernel file
- [ ] Create `fused_copy.py` Python wrapper
- [ ] Add to `kernel/__init__.py`

### Phase 3: Incremental Development
- [ ] Implement single copy (input_ids)
- [ ] Test and verify
- [ ] Add second copy (out_loc)
- [ ] Add remaining 1D copies
- [ ] Add 2D copy (page_table)
- [ ] Test all copies together

### Phase 4: Integration
- [ ] Integrate into `fa3.py`
- [ ] Add feature flag
- [ ] Test integration
- [ ] Verify correctness

### Phase 5: Optimization
- [ ] Tune kernel parameters
- [ ] Handle edge cases
- [ ] Benchmark vs. original
- [ ] Document results

## Key Design Decisions

### 1. Kernel Threading Strategy

**Option A: One thread per element**
- Simple to implement
- Good for small sizes
- May have overhead for many small copies

**Option B: Warp-based (32 threads)**
- More efficient for larger copies
- Better memory coalescing
- More complex indexing

**Recommendation:** Start with Option A, optimize to Option B if needed

### 2. Handling Variable Sizes

**Option A: Separate parameters for each size**
- Clear and explicit
- More parameters to pass

**Option B: Array of sizes**
- More flexible
- Slightly more complex

**Recommendation:** Start with Option A (explicit), can refactor later

### 3. 2D Tensor Handling

**Option A: Flatten to 1D**
- Simpler kernel code
- Same memory layout

**Option B: 2D indexing**
- More explicit
- Slightly more complex

**Recommendation:** Option A (flatten)

## Testing Strategy

### Unit Tests
1. **Correctness**: Verify all data copied correctly
2. **Edge Cases**: bs=1, bs=max, etc.
3. **Different Sizes**: Various batch sizes

### Integration Tests
1. **End-to-End**: Full decode with fused copy
2. **Multiple Replays**: Many consecutive replays
3. **Correctness**: Compare outputs vs. original

### Performance Tests
1. **Latency**: Time per replay
2. **Throughput**: Replays per second
3. **GPU Utilization**: During copies

## Success Criteria

1. **Correctness**: All tests pass, outputs match original
2. **Performance**: 20-40% reduction in copy overhead
3. **Reliability**: No crashes, handles edge cases
4. **Maintainability**: Clean, documented code

## Next Steps

1. **Start with Step 1.2**: Create benchmark baseline
2. **Then Step 2.1**: Create kernel file structure
3. **Then Step 3.1**: Implement single copy first
4. **Iterate**: Add one copy at a time, test each

## Questions to Answer During Implementation

1. What's the actual measured overhead? (Profile first)
2. Is warp-based threading better for our use case?
3. Should we handle non-contiguous tensors?
4. What's the optimal block size?
5. Should we support other backends (FlashInfer)?

