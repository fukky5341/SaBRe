## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 0.00327452442
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661)
1: (-0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296)
2: (-0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207)
3: (-0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227)
4: (-0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765)
5: (0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392)
6: (0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987)
7: (-0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057)
8: (-0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945)
9: (-0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784)

## BASE Result
execution time: IAR + LP analysis = 1.34 + 1.85 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0066965, upper bound: 0.0066965


# Binary Search by BASE starts (time budget: 2696.81 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=0.006939222104847431
rel_dist={5: [-0.005191087629474844, 0.005190667499944124]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.006939222104847431
rel_dist={5: [-0.00406464954046315, 0.004064320038150049]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.006702043581753969
rel_dist={5: [-0.00302959515626966, 0.0030295473171468856]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.006939222104847431
rel_dist={5: [-0.0035797029976590844, 0.0035794619380657977]}

## Binary Search Result
Binary search time: 17.07 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_dual_Z) starts
Time budget: 2679.74 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047486, upper bound: 0.0047483
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047486, upper bound: 0.0047483
time: 1.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.92
Output dim: 5, lower bound: -0.0047486, upper bound: 0.0047483
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.92
Output dim: 5, lower bound: -0.0047486, upper bound: 0.0047483

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047222, upper bound: 0.0046735
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046737, upper bound: 0.0047219
time: 0.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047222, upper bound: 0.0046735
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046737, upper bound: 0.0047219
time: 0.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -0.0047222, upper bound: 0.0046735
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -0.0046737, upper bound: 0.0047219
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -0.0047222, upper bound: 0.0046735
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -0.0046737, upper bound: 0.0047219

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113
time: 0.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0038115, upper bound: 0.0037927
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 5, lower bound: -0.0037929, upper bound: 0.0038113

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801
time: 1.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037804, upper bound: 0.0037519
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037748, upper bound: 0.0037616
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037619, upper bound: 0.0037747
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 5, lower bound: -0.0037521, upper bound: 0.0037801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457
time: 1.04 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037459, upper bound: 0.0037110
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037412, upper bound: 0.0037153
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037395, upper bound: 0.0037211
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037357, upper bound: 0.0037258
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037260, upper bound: 0.0037354
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037212, upper bound: 0.0037393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037409
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 5, lower bound: -0.0037112, upper bound: 0.0037457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037018
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037018
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037019
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037018
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428
time: 1.05 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037018
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037018
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037019
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037429, upper bound: 0.0036888
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037276, upper bound: 0.0037081
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037382, upper bound: 0.0036942
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037226, upper bound: 0.0037123
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037365, upper bound: 0.0037018
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037218, upper bound: 0.0037181
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037327, upper bound: 0.0037074
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037155, upper bound: 0.0037229
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037231, upper bound: 0.0037154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037075, upper bound: 0.0037326
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037182, upper bound: 0.0037215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037021, upper bound: 0.0037363
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037126, upper bound: 0.0037225
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036943, upper bound: 0.0037380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0037083, upper bound: 0.0037276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 5, lower bound: -0.0036890, upper bound: 0.0037428

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036273
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036274
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036643
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036642
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036273
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036274
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036643
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036642
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702
time: 1.07 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036273
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036274
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036643
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036642
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036273
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036703, upper bound: 0.0036118
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036528, upper bound: 0.0036152
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036553, upper bound: 0.0036274
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036348
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036657, upper bound: 0.0036172
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036488, upper bound: 0.0036206
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036501, upper bound: 0.0036306
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036341, upper bound: 0.0036393
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036643, upper bound: 0.0036247
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036452, upper bound: 0.0036278
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036489, upper bound: 0.0036386
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036319, upper bound: 0.0036439
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036611, upper bound: 0.0036297
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036414, upper bound: 0.0036337
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036432, upper bound: 0.0036425
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036264, upper bound: 0.0036488
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036643
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036490, upper bound: 0.0036261
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036426, upper bound: 0.0036430
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036340, upper bound: 0.0036411
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036299, upper bound: 0.0036609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036442, upper bound: 0.0036316
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036386, upper bound: 0.0036486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036281, upper bound: 0.0036450
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036249, upper bound: 0.0036642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036394, upper bound: 0.0036339
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036309, upper bound: 0.0036498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036209, upper bound: 0.0036488
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036174, upper bound: 0.0036655
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036350, upper bound: 0.0036397
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036276, upper bound: 0.0036551
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036154, upper bound: 0.0036525
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.76
Output dim: 5, lower bound: -0.0036120, upper bound: 0.0036702

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036431, upper bound: 0.0035852
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036417, upper bound: 0.0035853
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036260, upper bound: 0.0035884
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036260, upper bound: 0.0035885
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=0.006939222104847431
rel_dist={5: [-0.005463556997318264, 0.0054627970273385396]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041701, upper bound: 0.0041699
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041701, upper bound: 0.0041698
time: 1.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.37
Output dim: 5, lower bound: -0.0041701, upper bound: 0.0041699
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.37
Output dim: 5, lower bound: -0.0041701, upper bound: 0.0041698

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041540, upper bound: 0.0041189
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041191, upper bound: 0.0041537
time: 1.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041540, upper bound: 0.0041189
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041191, upper bound: 0.0041537
time: 1.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 5, lower bound: -0.0041540, upper bound: 0.0041189
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 5, lower bound: -0.0041191, upper bound: 0.0041537
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 5, lower bound: -0.0041540, upper bound: 0.0041189
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 5, lower bound: -0.0041191, upper bound: 0.0041537

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364
time: 1.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035367, upper bound: 0.0035237
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 5, lower bound: -0.0035240, upper bound: 0.0035364

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049
time: 1.15 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035053, upper bound: 0.0034882
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0035033, upper bound: 0.0034923
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034925, upper bound: 0.0035031
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.69
Output dim: 5, lower bound: -0.0034885, upper bound: 0.0035049

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034639
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034639
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034757
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034758
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034638
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034638
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034757
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034758
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813
time: 1.11 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034639
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034639
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034757
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034758
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034638
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034815, upper bound: 0.0034606
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034785, upper bound: 0.0034638
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034794, upper bound: 0.0034654
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034760, upper bound: 0.0034684
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034757
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034684, upper bound: 0.0034758
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034792
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034641, upper bound: 0.0034784
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 5, lower bound: -0.0034608, upper bound: 0.0034813

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034603
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034763
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034604
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034763
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034603
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034763
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034604
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034764
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783
time: 1.21 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034603
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034763
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034604
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034763
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034786, upper bound: 0.0034445
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034656, upper bound: 0.0034575
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034755, upper bound: 0.0034481
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034623, upper bound: 0.0034609
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034765, upper bound: 0.0034516
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034606, upper bound: 0.0034625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034730, upper bound: 0.0034542
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034572, upper bound: 0.0034652
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034603
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034763
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034654, upper bound: 0.0034571
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034544, upper bound: 0.0034728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034626, upper bound: 0.0034604
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034764
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034611, upper bound: 0.0034620
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034485, upper bound: 0.0034752
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034578, upper bound: 0.0034655
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 5, lower bound: -0.0034447, upper bound: 0.0034783

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034273, upper bound: 0.0033865
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034120, upper bound: 0.0033904
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034143, upper bound: 0.0033977
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034044
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034243, upper bound: 0.0033890
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034101, upper bound: 0.0033948
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034004
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033990, upper bound: 0.0034080
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034256, upper bound: 0.0033932
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034077, upper bound: 0.0033977
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034091, upper bound: 0.0034024
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033962, upper bound: 0.0034093
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034227, upper bound: 0.0033958
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034058, upper bound: 0.0034007
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034054
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033938, upper bound: 0.0034121
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034273, upper bound: 0.0033865
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034120, upper bound: 0.0033904
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034143, upper bound: 0.0033977
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034044
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034243, upper bound: 0.0033890
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034101, upper bound: 0.0033948
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034004
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033990, upper bound: 0.0034080
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034256, upper bound: 0.0033932
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034077, upper bound: 0.0033977
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034091, upper bound: 0.0034024
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033962, upper bound: 0.0034093
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034227, upper bound: 0.0033958
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034058, upper bound: 0.0034007
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034054
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033938, upper bound: 0.0034121
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034123, upper bound: 0.0033935
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034056
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034010, upper bound: 0.0034055
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033960, upper bound: 0.0034224
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034096, upper bound: 0.0033959
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034089
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033979, upper bound: 0.0034074
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033935, upper bound: 0.0034254
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034082, upper bound: 0.0033988
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034007, upper bound: 0.0034106
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033950, upper bound: 0.0034098
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033890, upper bound: 0.0034239
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034047, upper bound: 0.0034022
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033980, upper bound: 0.0034140
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033908, upper bound: 0.0034118
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033868, upper bound: 0.0034270
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034123, upper bound: 0.0033935
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034056
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034010, upper bound: 0.0034055
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033960, upper bound: 0.0034224
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034096, upper bound: 0.0033959
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034089
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033979, upper bound: 0.0034074
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033935, upper bound: 0.0034254
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034082, upper bound: 0.0033988
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034007, upper bound: 0.0034106
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033950, upper bound: 0.0034098
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033890, upper bound: 0.0034239
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034047, upper bound: 0.0034022
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033980, upper bound: 0.0034140
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033908, upper bound: 0.0034118
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033868, upper bound: 0.0034270
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034273, upper bound: 0.0033866
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034120, upper bound: 0.0033904
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034143, upper bound: 0.0033977
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034044
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034243, upper bound: 0.0033890
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034101, upper bound: 0.0033948
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034004
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033990, upper bound: 0.0034080
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034256, upper bound: 0.0033932
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034077, upper bound: 0.0033976
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034091, upper bound: 0.0034024
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033962, upper bound: 0.0034093
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034227, upper bound: 0.0033958
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034058, upper bound: 0.0034007
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034053
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033938, upper bound: 0.0034121
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034273, upper bound: 0.0033866
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034120, upper bound: 0.0033904
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034143, upper bound: 0.0033977
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034044
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034243, upper bound: 0.0033890
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034101, upper bound: 0.0033948
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034004
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033990, upper bound: 0.0034080
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034256, upper bound: 0.0033932
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034077, upper bound: 0.0033976
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034091, upper bound: 0.0034024
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033962, upper bound: 0.0034093
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034227, upper bound: 0.0033958
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034058, upper bound: 0.0034007
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034054
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033938, upper bound: 0.0034121
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034123, upper bound: 0.0033935
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034056
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034010, upper bound: 0.0034055
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033960, upper bound: 0.0034224
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034096, upper bound: 0.0033960
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034089
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033979, upper bound: 0.0034074
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033935, upper bound: 0.0034254
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034082, upper bound: 0.0033988
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034007, upper bound: 0.0034106
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033950, upper bound: 0.0034098
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033890, upper bound: 0.0034239
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034047, upper bound: 0.0034022
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033980, upper bound: 0.0034141
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033908, upper bound: 0.0034118
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033868, upper bound: 0.0034270
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034123, upper bound: 0.0033935
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034056, upper bound: 0.0034056
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034010, upper bound: 0.0034055
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033960, upper bound: 0.0034224
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034096, upper bound: 0.0033960
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034025, upper bound: 0.0034089
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033979, upper bound: 0.0034074
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033935, upper bound: 0.0034254
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034082, upper bound: 0.0033988
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034007, upper bound: 0.0034106
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.006939222104847431
rel_dist={5: [-0.004486465180227661, 0.004485917759426528]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034105
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034104
time: 1.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.24
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034105
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.24
Output dim: 5, lower bound: -0.0034108, upper bound: 0.0034104

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034028, upper bound: 0.0033805
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033808, upper bound: 0.0034026
time: 1.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034028, upper bound: 0.0033804
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033808, upper bound: 0.0034025
time: 1.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.04
Output dim: 5, lower bound: -0.0034028, upper bound: 0.0033805
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.04
Output dim: 5, lower bound: -0.0033808, upper bound: 0.0034026
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.04
Output dim: 5, lower bound: -0.0034028, upper bound: 0.0033804
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.04
Output dim: 5, lower bound: -0.0033808, upper bound: 0.0034025

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
time: 1.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030594, upper bound: 0.0030487
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.77
Output dim: 5, lower bound: -0.0030489, upper bound: 0.0030593
Binary search (step 2): status=Status.VERIFIED, k_low=2, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.006939222104847431
rel_dist={5: [-0.0035797029976590844, 0.0035794619380657977]}

## Binary search (step 3) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038305, upper bound: 0.0038302
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038305, upper bound: 0.0038302
time: 1.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.55
Output dim: 5, lower bound: -0.0038305, upper bound: 0.0038302
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.55
Output dim: 5, lower bound: -0.0038305, upper bound: 0.0038302

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038185, upper bound: 0.0037857
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037860, upper bound: 0.0038184
time: 1.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038185, upper bound: 0.0037857
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037860, upper bound: 0.0038184
time: 1.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 5, lower bound: -0.0038185, upper bound: 0.0037857
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 5, lower bound: -0.0037860, upper bound: 0.0038184
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 5, lower bound: -0.0038185, upper bound: 0.0037857
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 5, lower bound: -0.0037860, upper bound: 0.0038184

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322
time: 1.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033324, upper bound: 0.0033191
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041
time: 1.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033044, upper bound: 0.0032885
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0033029, upper bound: 0.0032916
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032917, upper bound: 0.0033027
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 5, lower bound: -0.0032887, upper bound: 0.0033041

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032673
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032673
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032673
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032671
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850
time: 1.23 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032673
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032673
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032673
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032853, upper bound: 0.0032671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032824, upper bound: 0.0032694
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032827, upper bound: 0.0032707
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032801
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032707, upper bound: 0.0032825
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032695, upper bound: 0.0032822
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 5, lower bound: -0.0032674, upper bound: 0.0032850

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032704
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032703
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032704
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032704
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822
time: 1.05 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032704
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032703
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032515
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0032643
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032795, upper bound: 0.0032555
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032672, upper bound: 0.0032664
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032797, upper bound: 0.0032578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032668, upper bound: 0.0032675
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032774, upper bound: 0.0032610
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032691
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032704
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032696, upper bound: 0.0032626
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032614, upper bound: 0.0032772
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032677, upper bound: 0.0032666
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032579, upper bound: 0.0032796
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032665, upper bound: 0.0032671
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032792
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032704
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 5, lower bound: -0.0032518, upper bound: 0.0032822

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015784

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
time: 1.29 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032442, upper bound: 0.0032079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032297, upper bound: 0.0032106
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032416, upper bound: 0.0032113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032275, upper bound: 0.0032149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032419, upper bound: 0.0032136
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032267, upper bound: 0.0032170
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032395, upper bound: 0.0032170
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032243, upper bound: 0.0032207
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032210, upper bound: 0.0032241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032172, upper bound: 0.0032389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032174, upper bound: 0.0032263
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032138, upper bound: 0.0032416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032151, upper bound: 0.0032274
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032115, upper bound: 0.0032410
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032111, upper bound: 0.0032295
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.19
Output dim: 5, lower bound: -0.0032082, upper bound: 0.0032437
Binary search (step 3): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=0.006939222104847431
rel_dist={5: [-0.00406464954046315, 0.004064320038150049]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1666.34 seconds
