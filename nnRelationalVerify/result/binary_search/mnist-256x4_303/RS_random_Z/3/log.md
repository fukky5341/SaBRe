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
execution time: IAR + LP analysis = 1.31 + 1.86 = 3.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0066965, upper bound: 0.0066965


# Binary Search by BASE starts (time budget: 2696.83 seconds, max iter: 100)

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
Binary search time: 17.02 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_random_Z) starts
Time budget: 2679.81 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0051911, upper bound: 0.0051905
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0051911, upper bound: 0.0051903
time: 1.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.12
Output dim: 5, lower bound: -0.0051911, upper bound: 0.0051905
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.12
Output dim: 5, lower bound: -0.0051911, upper bound: 0.0051903

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0051691, upper bound: 0.0051036
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0051039, upper bound: 0.0051688
time: 1.63 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048908, upper bound: 0.0048905
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048908, upper bound: 0.0048905
time: 1.42 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0051691, upper bound: 0.0051036
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0051039, upper bound: 0.0051688
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0048908, upper bound: 0.0048905
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0048908, upper bound: 0.0048905

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0049511, upper bound: 0.0048815
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0049503, upper bound: 0.0048829
time: 1.53 seconds

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0050263, upper bound: 0.0050726
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0050128, upper bound: 0.0050856
time: 1.68 seconds

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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048355, upper bound: 0.0048234
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048237, upper bound: 0.0048353
time: 1.57 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046973, upper bound: 0.0046754
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046755, upper bound: 0.0046970
time: 1.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0049511, upper bound: 0.0048815
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0049503, upper bound: 0.0048829
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0050263, upper bound: 0.0050726
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0050128, upper bound: 0.0050856
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0048355, upper bound: 0.0048234
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0048237, upper bound: 0.0048353
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0046973, upper bound: 0.0046754
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 5, lower bound: -0.0046755, upper bound: 0.0046970

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043345, upper bound: 0.0042923
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043345, upper bound: 0.0042923
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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048089, upper bound: 0.0047497
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048080, upper bound: 0.0047498
time: 1.71 seconds

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0049437, upper bound: 0.0049644
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0049274, upper bound: 0.0049943
time: 1.53 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040887, upper bound: 0.0041280
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040887, upper bound: 0.0041280
time: 1.26 seconds

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048168, upper bound: 0.0048050
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048171, upper bound: 0.0048028
time: 1.51 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043242, upper bound: 0.0043250
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043242, upper bound: 0.0043250
time: 1.31 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046671, upper bound: 0.0046440
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046643, upper bound: 0.0046459
time: 1.44 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046721, upper bound: 0.0046216
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045959, upper bound: 0.0046933
time: 1.49 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0043345, upper bound: 0.0042923
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0043345, upper bound: 0.0042923
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0048089, upper bound: 0.0047497
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0048080, upper bound: 0.0047498
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0049437, upper bound: 0.0049644
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0049274, upper bound: 0.0049943
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0040887, upper bound: 0.0041280
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0040887, upper bound: 0.0041280
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0048168, upper bound: 0.0048050
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0048171, upper bound: 0.0048028
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0043242, upper bound: 0.0043250
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0043242, upper bound: 0.0043250
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0046671, upper bound: 0.0046440
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0046643, upper bound: 0.0046459
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0046721, upper bound: 0.0046216
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 5, lower bound: -0.0045959, upper bound: 0.0046933

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043000, upper bound: 0.0042296
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042774, upper bound: 0.0042567
time: 1.32 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041124, upper bound: 0.0040800
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041124, upper bound: 0.0040800
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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045529, upper bound: 0.0044962
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045529, upper bound: 0.0044962
time: 1.46 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047569, upper bound: 0.0046832
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047479, upper bound: 0.0046976
time: 1.62 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0049405, upper bound: 0.0048789
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048579, upper bound: 0.0049612
time: 1.62 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048888, upper bound: 0.0049316
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048666, upper bound: 0.0049563
time: 1.75 seconds

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040840, upper bound: 0.0040947
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040668, upper bound: 0.0041232
time: 1.29 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035640, upper bound: 0.0035759
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035640, upper bound: 0.0035758
time: 1.07 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047956, upper bound: 0.0047776
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047912, upper bound: 0.0047835
time: 1.51 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048132, upper bound: 0.0047304
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047370, upper bound: 0.0047989
time: 1.51 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042624, upper bound: 0.0042140
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042123, upper bound: 0.0042630
time: 1.32 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042393, upper bound: 0.0042409
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042393, upper bound: 0.0042409
time: 1.31 seconds

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
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045776, upper bound: 0.0045609
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045776, upper bound: 0.0045609
time: 1.68 seconds

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046252, upper bound: 0.0045860
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046013, upper bound: 0.0046066
time: 1.47 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046254, upper bound: 0.0045092
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044992, upper bound: 0.0045677
time: 1.54 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045155, upper bound: 0.0045815
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044896, upper bound: 0.0046112
time: 1.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0043000, upper bound: 0.0042296
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0042774, upper bound: 0.0042567
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0041124, upper bound: 0.0040800
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0041124, upper bound: 0.0040800
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0045529, upper bound: 0.0044962
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0045529, upper bound: 0.0044962
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0047569, upper bound: 0.0046832
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0047479, upper bound: 0.0046976
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0049405, upper bound: 0.0048789
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0048579, upper bound: 0.0049612
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0048888, upper bound: 0.0049316
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0048666, upper bound: 0.0049563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0040840, upper bound: 0.0040947
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0040668, upper bound: 0.0041232
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0035640, upper bound: 0.0035759
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0035640, upper bound: 0.0035758
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0047956, upper bound: 0.0047776
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0047912, upper bound: 0.0047835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0048132, upper bound: 0.0047304
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0047370, upper bound: 0.0047989
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0042624, upper bound: 0.0042140
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0042123, upper bound: 0.0042630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0042393, upper bound: 0.0042409
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0042393, upper bound: 0.0042409
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0045776, upper bound: 0.0045609
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0045776, upper bound: 0.0045609
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0046252, upper bound: 0.0045860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0046013, upper bound: 0.0046066
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0046254, upper bound: 0.0045092
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0044992, upper bound: 0.0045677
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0045155, upper bound: 0.0045815
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 5, lower bound: -0.0044896, upper bound: 0.0046112

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042301, upper bound: 0.0041579
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042306, upper bound: 0.0041587
time: 1.33 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040631, upper bound: 0.0040476
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040631, upper bound: 0.0040476
time: 1.31 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040600, upper bound: 0.0040213
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040565, upper bound: 0.0040276
time: 1.35 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040765, upper bound: 0.0040434
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040764, upper bound: 0.0040437
time: 1.30 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045494, upper bound: 0.0044442
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044927, upper bound: 0.0044933
time: 1.57 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045224, upper bound: 0.0044577
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045184, upper bound: 0.0044670
time: 1.45 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046764, upper bound: 0.0045863
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046414, upper bound: 0.0045994
time: 1.58 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046484, upper bound: 0.0045987
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046484, upper bound: 0.0045989
time: 1.76 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037173, upper bound: 0.0037287
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037173, upper bound: 0.0037287
time: 1.08 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036343, upper bound: 0.0036480
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036343, upper bound: 0.0036480
time: 1.12 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048351, upper bound: 0.0047341
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046793, upper bound: 0.0048768
time: 1.63 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048225, upper bound: 0.0048879
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048107, upper bound: 0.0049049
time: 1.70 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038063, upper bound: 0.0038178
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038063, upper bound: 0.0038178
time: 1.23 seconds

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
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035703, upper bound: 0.0036147
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035703, upper bound: 0.0036147
time: 1.00 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035094, upper bound: 0.0035215
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035094, upper bound: 0.0035215
time: 1.05 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032838, upper bound: 0.0032849
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032838, upper bound: 0.0032849
time: 0.93 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037440, upper bound: 0.0037317
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037440, upper bound: 0.0037317
time: 1.11 seconds

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047523, upper bound: 0.0047223
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047288, upper bound: 0.0047438
time: 1.50 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047805, upper bound: 0.0046979
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047756, upper bound: 0.0046978
time: 1.65 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046962, upper bound: 0.0047359
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046832, upper bound: 0.0047597
time: 1.66 seconds

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
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040351, upper bound: 0.0039891
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040304, upper bound: 0.0039905
time: 1.33 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042084, upper bound: 0.0042201
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041779, upper bound: 0.0042588
time: 1.07 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041845, upper bound: 0.0041737
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041743, upper bound: 0.0041840
time: 1.29 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040208, upper bound: 0.0040152
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040143, upper bound: 0.0040220
time: 1.35 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044103, upper bound: 0.0043982
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044103, upper bound: 0.0043982
time: 1.62 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039987, upper bound: 0.0039950
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039987, upper bound: 0.0039950
time: 1.42 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045706, upper bound: 0.0044588
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044927, upper bound: 0.0045291
time: 1.61 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042888, upper bound: 0.0043022
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042888, upper bound: 0.0043018
time: 1.51 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045529, upper bound: 0.0044354
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045535, upper bound: 0.0044375
time: 1.64 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043837, upper bound: 0.0044785
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043582, upper bound: 0.0044805
time: 1.57 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044598, upper bound: 0.0044559
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044054, upper bound: 0.0045264
time: 1.59 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039455, upper bound: 0.0040011
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039455, upper bound: 0.0040007
time: 1.43 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0042301, upper bound: 0.0041579
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0042306, upper bound: 0.0041587
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040631, upper bound: 0.0040476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040631, upper bound: 0.0040476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040600, upper bound: 0.0040213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040565, upper bound: 0.0040276
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040765, upper bound: 0.0040434
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040764, upper bound: 0.0040437
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0045494, upper bound: 0.0044442
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0044927, upper bound: 0.0044933
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0045224, upper bound: 0.0044577
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0045184, upper bound: 0.0044670
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0046764, upper bound: 0.0045863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0046414, upper bound: 0.0045994
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0046484, upper bound: 0.0045987
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0046484, upper bound: 0.0045989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0037173, upper bound: 0.0037287
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0037173, upper bound: 0.0037287
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0036343, upper bound: 0.0036480
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0036343, upper bound: 0.0036480
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0048351, upper bound: 0.0047341
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0046793, upper bound: 0.0048768
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0048225, upper bound: 0.0048879
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0048107, upper bound: 0.0049049
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0038063, upper bound: 0.0038178
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0038063, upper bound: 0.0038178
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0035703, upper bound: 0.0036147
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0035703, upper bound: 0.0036147
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0035094, upper bound: 0.0035215
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0035094, upper bound: 0.0035215
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0032838, upper bound: 0.0032849
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0032838, upper bound: 0.0032849
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0037440, upper bound: 0.0037317
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0037440, upper bound: 0.0037317
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0047523, upper bound: 0.0047223
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0047288, upper bound: 0.0047438
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0047805, upper bound: 0.0046979
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0047756, upper bound: 0.0046978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0046962, upper bound: 0.0047359
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0046832, upper bound: 0.0047597
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040351, upper bound: 0.0039891
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040304, upper bound: 0.0039905
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0042084, upper bound: 0.0042201
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0041779, upper bound: 0.0042588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0041845, upper bound: 0.0041737
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0041743, upper bound: 0.0041840
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040208, upper bound: 0.0040152
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0040143, upper bound: 0.0040220
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0044103, upper bound: 0.0043982
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0044103, upper bound: 0.0043982
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0039987, upper bound: 0.0039950
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0039987, upper bound: 0.0039950
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0045706, upper bound: 0.0044588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0044927, upper bound: 0.0045291
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0042888, upper bound: 0.0043022
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0042888, upper bound: 0.0043018
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0045529, upper bound: 0.0044354
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0045535, upper bound: 0.0044375
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0043837, upper bound: 0.0044785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0043582, upper bound: 0.0044805
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0044598, upper bound: 0.0044559
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0044054, upper bound: 0.0045264
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0039455, upper bound: 0.0040011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 5, lower bound: -0.0039455, upper bound: 0.0040007

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041780, upper bound: 0.0040686
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041486, upper bound: 0.0041062
time: 1.37 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042142, upper bound: 0.0040766
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041502, upper bound: 0.0041423
time: 1.31 seconds

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039969, upper bound: 0.0039770
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039969, upper bound: 0.0039770
time: 1.43 seconds

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037793, upper bound: 0.0037701
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037793, upper bound: 0.0037701
time: 1.24 seconds

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040313, upper bound: 0.0039932
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040293, upper bound: 0.0039932
time: 1.29 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040209, upper bound: 0.0039908
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040205, upper bound: 0.0039906
time: 1.35 seconds

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040593, upper bound: 0.0039815
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040160, upper bound: 0.0040264
time: 1.38 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040453, upper bound: 0.0040029
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040410, upper bound: 0.0040133
time: 1.30 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045308, upper bound: 0.0044239
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045299, upper bound: 0.0044253
time: 1.62 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043018, upper bound: 0.0042790
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042795, upper bound: 0.0043067
time: 1.61 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044943, upper bound: 0.0044279
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044938, upper bound: 0.0044292
time: 1.58 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044622, upper bound: 0.0044006
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044534, upper bound: 0.0044139
time: 1.50 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046547, upper bound: 0.0045541
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046378, upper bound: 0.0045674
time: 1.69 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045443, upper bound: 0.0044980
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045443, upper bound: 0.0044980
time: 1.86 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045947, upper bound: 0.0043902
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044474, upper bound: 0.0045448
time: 1.67 seconds

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046280, upper bound: 0.0045562
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046125, upper bound: 0.0045779
time: 1.70 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036607, upper bound: 0.0036300
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036081, upper bound: 0.0036710
time: 0.96 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036681, upper bound: 0.0036799
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036682, upper bound: 0.0036785
time: 1.16 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036126, upper bound: 0.0036220
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036089, upper bound: 0.0036247
time: 0.92 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035708, upper bound: 0.0035580
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035536, upper bound: 0.0035866
time: 1.09 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047938, upper bound: 0.0046766
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047764, upper bound: 0.0046901
time: 1.73 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046086, upper bound: 0.0048067
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046086, upper bound: 0.0048050
time: 1.72 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0048079, upper bound: 0.0047558
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046823, upper bound: 0.0048734
time: 1.88 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047919, upper bound: 0.0048841
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047910, upper bound: 0.0048854
time: 1.72 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037687, upper bound: 0.0037744
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037626, upper bound: 0.0037796
time: 1.30 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033077, upper bound: 0.0033143
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033077, upper bound: 0.0033143
time: 0.95 seconds

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
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035100, upper bound: 0.0035425
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034917, upper bound: 0.0035546
time: 1.14 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035007, upper bound: 0.0035438
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035007, upper bound: 0.0035442
time: 0.91 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034779, upper bound: 0.0034893
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034773, upper bound: 0.0034899
time: 1.05 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034338, upper bound: 0.0034388
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034338, upper bound: 0.0034388
time: 1.03 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032495, upper bound: 0.0032357
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032352, upper bound: 0.0032507
time: 0.97 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032144, upper bound: 0.0032150
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032141, upper bound: 0.0032153
time: 0.90 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037278, upper bound: 0.0036720
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036851, upper bound: 0.0037159
time: 1.14 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037207, upper bound: 0.0036967
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037123, upper bound: 0.0037098
time: 1.09 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047378, upper bound: 0.0046085
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046285, upper bound: 0.0047080
time: 1.52 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046824, upper bound: 0.0045753
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045560, upper bound: 0.0046968
time: 1.53 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047579, upper bound: 0.0046272
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046953, upper bound: 0.0046764
time: 1.61 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0047041, upper bound: 0.0046211
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046953, upper bound: 0.0046329
time: 1.69 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041395, upper bound: 0.0041396
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041395, upper bound: 0.0041396
time: 1.31 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046622, upper bound: 0.0046730
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0046126, upper bound: 0.0047383
time: 1.55 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039736, upper bound: 0.0039238
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039642, upper bound: 0.0039327
time: 1.33 seconds

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037563, upper bound: 0.0037264
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037563, upper bound: 0.0037264
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041922, upper bound: 0.0041213
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041298, upper bound: 0.0042041
time: 1.22 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034533, upper bound: 0.0034883
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034533, upper bound: 0.0034883
time: 1.07 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034528, upper bound: 0.0034438
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034528, upper bound: 0.0034438
time: 1.09 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034440, upper bound: 0.0034516
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034440, upper bound: 0.0034516
time: 1.08 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039952, upper bound: 0.0039527
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039611, upper bound: 0.0039894
time: 1.34 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039413, upper bound: 0.0039338
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039246, upper bound: 0.0039466
time: 1.52 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039674, upper bound: 0.0039599
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039674, upper bound: 0.0039599
time: 1.36 seconds

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
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043177, upper bound: 0.0042997
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043137, upper bound: 0.0043048
time: 1.60 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039637, upper bound: 0.0039547
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039624, upper bound: 0.0039598
time: 1.27 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039099, upper bound: 0.0038914
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038975, upper bound: 0.0039075
time: 1.43 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044779, upper bound: 0.0043693
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044707, upper bound: 0.0043767
time: 1.63 seconds

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039169, upper bound: 0.0039355
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039169, upper bound: 0.0039355
time: 1.41 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040895, upper bound: 0.0041090
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040895, upper bound: 0.0041090
time: 1.55 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042588, upper bound: 0.0042721
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042574, upper bound: 0.0042727
time: 1.52 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042525, upper bound: 0.0041496
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042525, upper bound: 0.0041496
time: 1.56 seconds

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045201, upper bound: 0.0044016
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0045154, upper bound: 0.0044035
time: 1.19 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043047, upper bound: 0.0043808
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042862, upper bound: 0.0043958
time: 1.61 seconds

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043062, upper bound: 0.0044234
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043045, upper bound: 0.0044273
time: 1.59 seconds

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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044378, upper bound: 0.0043845
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043872, upper bound: 0.0044317
time: 1.61 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043417, upper bound: 0.0044408
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043297, upper bound: 0.0044489
time: 1.42 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039095, upper bound: 0.0039662
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039099, upper bound: 0.0039657
time: 1.30 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038228, upper bound: 0.0038803
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038228, upper bound: 0.0038802
time: 1.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0041780, upper bound: 0.0040686
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0041486, upper bound: 0.0041062
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0042142, upper bound: 0.0040766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0041502, upper bound: 0.0041423
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039969, upper bound: 0.0039770
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039969, upper bound: 0.0039770
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037793, upper bound: 0.0037701
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037793, upper bound: 0.0037701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040313, upper bound: 0.0039932
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040293, upper bound: 0.0039932
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040209, upper bound: 0.0039908
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040205, upper bound: 0.0039906
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040593, upper bound: 0.0039815
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040160, upper bound: 0.0040264
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040453, upper bound: 0.0040029
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040410, upper bound: 0.0040133
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045308, upper bound: 0.0044239
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045299, upper bound: 0.0044253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043018, upper bound: 0.0042790
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0042795, upper bound: 0.0043067
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044943, upper bound: 0.0044279
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044938, upper bound: 0.0044292
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044622, upper bound: 0.0044006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044534, upper bound: 0.0044139
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046547, upper bound: 0.0045541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046378, upper bound: 0.0045674
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045443, upper bound: 0.0044980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045443, upper bound: 0.0044980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045947, upper bound: 0.0043902
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044474, upper bound: 0.0045448
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046280, upper bound: 0.0045562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046125, upper bound: 0.0045779
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0036607, upper bound: 0.0036300
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0036081, upper bound: 0.0036710
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0036681, upper bound: 0.0036799
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0036682, upper bound: 0.0036785
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0036126, upper bound: 0.0036220
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0036089, upper bound: 0.0036247
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0035708, upper bound: 0.0035580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0035536, upper bound: 0.0035866
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0047938, upper bound: 0.0046766
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0047764, upper bound: 0.0046901
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046086, upper bound: 0.0048067
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046086, upper bound: 0.0048050
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0048079, upper bound: 0.0047558
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046823, upper bound: 0.0048734
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0047919, upper bound: 0.0048841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0047910, upper bound: 0.0048854
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037687, upper bound: 0.0037744
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037626, upper bound: 0.0037796
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0033077, upper bound: 0.0033143
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0033077, upper bound: 0.0033143
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0035100, upper bound: 0.0035425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034917, upper bound: 0.0035546
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0035007, upper bound: 0.0035438
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0035007, upper bound: 0.0035442
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034779, upper bound: 0.0034893
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034773, upper bound: 0.0034899
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034338, upper bound: 0.0034388
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034338, upper bound: 0.0034388
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0032495, upper bound: 0.0032357
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0032352, upper bound: 0.0032507
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0032144, upper bound: 0.0032150
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0032141, upper bound: 0.0032153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037278, upper bound: 0.0036720
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0036851, upper bound: 0.0037159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037207, upper bound: 0.0036967
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037123, upper bound: 0.0037098
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0047378, upper bound: 0.0046085
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046285, upper bound: 0.0047080
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046824, upper bound: 0.0045753
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045560, upper bound: 0.0046968
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0047579, upper bound: 0.0046272
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046953, upper bound: 0.0046764
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0047041, upper bound: 0.0046211
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046953, upper bound: 0.0046329
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0041395, upper bound: 0.0041396
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0041395, upper bound: 0.0041396
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046622, upper bound: 0.0046730
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0046126, upper bound: 0.0047383
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039736, upper bound: 0.0039238
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039642, upper bound: 0.0039327
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037563, upper bound: 0.0037264
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0037563, upper bound: 0.0037264
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0041922, upper bound: 0.0041213
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0041298, upper bound: 0.0042041
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034533, upper bound: 0.0034883
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034533, upper bound: 0.0034883
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034528, upper bound: 0.0034438
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034528, upper bound: 0.0034438
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034440, upper bound: 0.0034516
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0034440, upper bound: 0.0034516
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039952, upper bound: 0.0039527
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039611, upper bound: 0.0039894
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039413, upper bound: 0.0039338
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039246, upper bound: 0.0039466
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039674, upper bound: 0.0039599
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039674, upper bound: 0.0039599
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043177, upper bound: 0.0042997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043137, upper bound: 0.0043048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039637, upper bound: 0.0039547
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039624, upper bound: 0.0039598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039099, upper bound: 0.0038914
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0038975, upper bound: 0.0039075
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044779, upper bound: 0.0043693
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044707, upper bound: 0.0043767
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039169, upper bound: 0.0039355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039169, upper bound: 0.0039355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040895, upper bound: 0.0041090
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0040895, upper bound: 0.0041090
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0042588, upper bound: 0.0042721
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0042574, upper bound: 0.0042727
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0042525, upper bound: 0.0041496
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0042525, upper bound: 0.0041496
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045201, upper bound: 0.0044016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0045154, upper bound: 0.0044035
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043047, upper bound: 0.0043808
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0042862, upper bound: 0.0043958
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043062, upper bound: 0.0044234
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043045, upper bound: 0.0044273
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0044378, upper bound: 0.0043845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043872, upper bound: 0.0044317
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043417, upper bound: 0.0044408
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0043297, upper bound: 0.0044489
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039095, upper bound: 0.0039662
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0039099, upper bound: 0.0039657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0038228, upper bound: 0.0038803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.98
Output dim: 5, lower bound: -0.0038228, upper bound: 0.0038802

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041216, upper bound: 0.0038987
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040019, upper bound: 0.0040126
time: 1.47 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040923, upper bound: 0.0039537
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039617, upper bound: 0.0040493
time: 1.33 seconds

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

Time for backsubstitution: 1.29 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=0.006939222104847431
rel_dist={5: [-0.005463556997318264, 0.0054627970273385396]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0044158, upper bound: 0.0043635
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043639, upper bound: 0.0044154
time: 1.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.61
Output dim: 5, lower bound: -0.0044158, upper bound: 0.0043635
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.61
Output dim: 5, lower bound: -0.0043639, upper bound: 0.0044154

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043858, upper bound: 0.0043319
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043836, upper bound: 0.0043338
time: 1.89 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043425, upper bound: 0.0043906
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043401, upper bound: 0.0043939
time: 1.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.93
Output dim: 5, lower bound: -0.0043858, upper bound: 0.0043319
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.93
Output dim: 5, lower bound: -0.0043836, upper bound: 0.0043338
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.93
Output dim: 5, lower bound: -0.0043425, upper bound: 0.0043906
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.93
Output dim: 5, lower bound: -0.0043401, upper bound: 0.0043939

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042736, upper bound: 0.0041913
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042441, upper bound: 0.0042221
time: 1.87 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040976, upper bound: 0.0040868
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040976, upper bound: 0.0040863
time: 1.79 seconds

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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043281, upper bound: 0.0043363
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042901, upper bound: 0.0043766
time: 1.94 seconds

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0043144, upper bound: 0.0043477
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042974, upper bound: 0.0043699
time: 1.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0042736, upper bound: 0.0041913
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0042441, upper bound: 0.0042221
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0040976, upper bound: 0.0040868
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0040976, upper bound: 0.0040863
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0043281, upper bound: 0.0043363
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0042901, upper bound: 0.0043766
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0043144, upper bound: 0.0043477
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 5, lower bound: -0.0042974, upper bound: 0.0043699

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042596, upper bound: 0.0041420
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042171, upper bound: 0.0041766
time: 1.80 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041968, upper bound: 0.0041769
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041968, upper bound: 0.0041770
time: 2.10 seconds

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040923, upper bound: 0.0040200
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040296, upper bound: 0.0040815
time: 1.70 seconds

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040770, upper bound: 0.0040649
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040757, upper bound: 0.0040659
time: 1.76 seconds

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042884, upper bound: 0.0042708
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042555, upper bound: 0.0042956
time: 1.97 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042459, upper bound: 0.0043288
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0042459, upper bound: 0.0043288
time: 1.87 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041744, upper bound: 0.0041896
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041743, upper bound: 0.0041895
time: 1.45 seconds

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034087, upper bound: 0.0034087
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034087, upper bound: 0.0034087
time: 0.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0042596, upper bound: 0.0041420
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0042171, upper bound: 0.0041766
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0041968, upper bound: 0.0041769
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0041968, upper bound: 0.0041770
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0040923, upper bound: 0.0040200
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0040296, upper bound: 0.0040815
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0040770, upper bound: 0.0040649
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0040757, upper bound: 0.0040659
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0042884, upper bound: 0.0042708
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0042555, upper bound: 0.0042956
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0042459, upper bound: 0.0043288
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0042459, upper bound: 0.0043288
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0041744, upper bound: 0.0041896
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0041743, upper bound: 0.0041895
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0034087, upper bound: 0.0034087
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 5, lower bound: -0.0034087, upper bound: 0.0034087

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041934, upper bound: 0.0040786
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041918, upper bound: 0.0040781
time: 1.39 seconds

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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041941, upper bound: 0.0041351
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041668, upper bound: 0.0041498
time: 1.90 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041806, upper bound: 0.0040791
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041000, upper bound: 0.0041611
time: 1.60 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041518, upper bound: 0.0041135
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041423, upper bound: 0.0041307
time: 1.92 seconds

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040491, upper bound: 0.0039656
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040289, upper bound: 0.0039734
time: 1.61 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039872, upper bound: 0.0040313
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039859, upper bound: 0.0040374
time: 1.27 seconds

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033513, upper bound: 0.0033529
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033513, upper bound: 0.0033529
time: 1.10 seconds

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039698, upper bound: 0.0039612
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039698, upper bound: 0.0039612
time: 1.53 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041408, upper bound: 0.0041185
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041408, upper bound: 0.0041185
time: 1.64 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041979, upper bound: 0.0042113
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041816, upper bound: 0.0042367
time: 1.90 seconds

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041952, upper bound: 0.0042402
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041771, upper bound: 0.0042803
time: 1.94 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041573, upper bound: 0.0042300
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041545, upper bound: 0.0042406
time: 1.92 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037449, upper bound: 0.0037448
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037449, upper bound: 0.0037448
time: 1.41 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041062, upper bound: 0.0041190
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041063, upper bound: 0.0041186
time: 1.83 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033629, upper bound: 0.0033569
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033578, upper bound: 0.0033627
time: 0.95 seconds

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033420, upper bound: 0.0033058
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033027, upper bound: 0.0033421
time: 1.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041934, upper bound: 0.0040786
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041918, upper bound: 0.0040781
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041941, upper bound: 0.0041351
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041668, upper bound: 0.0041498
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041806, upper bound: 0.0040791
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041000, upper bound: 0.0041611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041518, upper bound: 0.0041135
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041423, upper bound: 0.0041307
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0040491, upper bound: 0.0039656
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0040289, upper bound: 0.0039734
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0039872, upper bound: 0.0040313
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0039859, upper bound: 0.0040374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0033513, upper bound: 0.0033529
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0033513, upper bound: 0.0033529
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0039698, upper bound: 0.0039612
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0039698, upper bound: 0.0039612
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041408, upper bound: 0.0041185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041408, upper bound: 0.0041185
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041979, upper bound: 0.0042113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041816, upper bound: 0.0042367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041952, upper bound: 0.0042402
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041771, upper bound: 0.0042803
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041573, upper bound: 0.0042300
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041545, upper bound: 0.0042406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0037449, upper bound: 0.0037448
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0037449, upper bound: 0.0037448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041062, upper bound: 0.0041190
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0041063, upper bound: 0.0041186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0033629, upper bound: 0.0033569
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0033578, upper bound: 0.0033627
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0033420, upper bound: 0.0033058
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.55
Output dim: 5, lower bound: -0.0033027, upper bound: 0.0033421

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041454, upper bound: 0.0039249
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040312, upper bound: 0.0040272
time: 1.36 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041861, upper bound: 0.0039965
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041249, upper bound: 0.0040723
time: 1.94 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038735, upper bound: 0.0038740
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038735, upper bound: 0.0038735
time: 1.75 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041236, upper bound: 0.0040912
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041075, upper bound: 0.0041037
time: 1.93 seconds

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041392, upper bound: 0.0040095
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041190, upper bound: 0.0040385
time: 1.71 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037545, upper bound: 0.0038302
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037545, upper bound: 0.0038302
time: 1.83 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041135, upper bound: 0.0040710
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041075, upper bound: 0.0040753
time: 1.85 seconds

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
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032353, upper bound: 0.0032369
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032353, upper bound: 0.0032369
time: 1.14 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040037, upper bound: 0.0038699
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039575, upper bound: 0.0039217
time: 1.70 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040130, upper bound: 0.0038770
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039327, upper bound: 0.0039576
time: 1.66 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039514, upper bound: 0.0039893
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039499, upper bound: 0.0039950
time: 1.67 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036302, upper bound: 0.0036667
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036302, upper bound: 0.0036667
time: 1.41 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033462, upper bound: 0.0033293
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033281, upper bound: 0.0033476
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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031286, upper bound: 0.0031283
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031286, upper bound: 0.0031283
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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037723, upper bound: 0.0037725
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037723, upper bound: 0.0037725
time: 1.59 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036693, upper bound: 0.0036673
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036693, upper bound: 0.0036675
time: 1.43 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040972, upper bound: 0.0040305
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040471, upper bound: 0.0040738
time: 1.79 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040997, upper bound: 0.0040646
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040871, upper bound: 0.0040785
time: 1.82 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036247, upper bound: 0.0036218
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036247, upper bound: 0.0036218
time: 1.46 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040167, upper bound: 0.0040236
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040167, upper bound: 0.0040235
time: 1.71 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041067, upper bound: 0.0041421
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041034, upper bound: 0.0041527
time: 1.90 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041015, upper bound: 0.0042022
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041015, upper bound: 0.0042016
time: 1.99 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039488, upper bound: 0.0039902
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039488, upper bound: 0.0039903
time: 1.71 seconds

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041001, upper bound: 0.0040779
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040067, upper bound: 0.0041891
time: 2.05 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037056, upper bound: 0.0037011
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037012, upper bound: 0.0037059
time: 1.42 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035923, upper bound: 0.0035929
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035923, upper bound: 0.0035929
time: 1.48 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039474, upper bound: 0.0039496
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039474, upper bound: 0.0039514
time: 1.57 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039924, upper bound: 0.0039902
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039860, upper bound: 0.0039947
time: 1.76 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033432, upper bound: 0.0033176
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033249, upper bound: 0.0033373
time: 1.07 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033095, upper bound: 0.0033015
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032966, upper bound: 0.0033143
time: 1.27 seconds

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032830, upper bound: 0.0032483
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032788, upper bound: 0.0032505
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032893
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032500, upper bound: 0.0032952
time: 1.10 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041454, upper bound: 0.0039249
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040312, upper bound: 0.0040272
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041861, upper bound: 0.0039965
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041249, upper bound: 0.0040723
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0038735, upper bound: 0.0038740
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0038735, upper bound: 0.0038735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041236, upper bound: 0.0040912
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041075, upper bound: 0.0041037
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041392, upper bound: 0.0040095
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041190, upper bound: 0.0040385
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0037545, upper bound: 0.0038302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0037545, upper bound: 0.0038302
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041135, upper bound: 0.0040710
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041075, upper bound: 0.0040753
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0032353, upper bound: 0.0032369
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0032353, upper bound: 0.0032369
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040037, upper bound: 0.0038699
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039575, upper bound: 0.0039217
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040130, upper bound: 0.0038770
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039327, upper bound: 0.0039576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039514, upper bound: 0.0039893
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039499, upper bound: 0.0039950
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0036302, upper bound: 0.0036667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0036302, upper bound: 0.0036667
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0033462, upper bound: 0.0033293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0033281, upper bound: 0.0033476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0031286, upper bound: 0.0031283
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0031286, upper bound: 0.0031283
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0037723, upper bound: 0.0037725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0037723, upper bound: 0.0037725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0036693, upper bound: 0.0036673
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0036693, upper bound: 0.0036675
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040972, upper bound: 0.0040305
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040471, upper bound: 0.0040738
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040997, upper bound: 0.0040646
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040871, upper bound: 0.0040785
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0036247, upper bound: 0.0036218
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0036247, upper bound: 0.0036218
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040167, upper bound: 0.0040236
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040167, upper bound: 0.0040235
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041067, upper bound: 0.0041421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041034, upper bound: 0.0041527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041015, upper bound: 0.0042022
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041015, upper bound: 0.0042016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039488, upper bound: 0.0039902
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039488, upper bound: 0.0039903
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0041001, upper bound: 0.0040779
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0040067, upper bound: 0.0041891
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0037056, upper bound: 0.0037011
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0037012, upper bound: 0.0037059
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0035923, upper bound: 0.0035929
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0035923, upper bound: 0.0035929
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039474, upper bound: 0.0039496
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039474, upper bound: 0.0039514
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039924, upper bound: 0.0039902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0039860, upper bound: 0.0039947
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0033432, upper bound: 0.0033176
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0033249, upper bound: 0.0033373
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0033095, upper bound: 0.0033015
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0032966, upper bound: 0.0033143
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0032830, upper bound: 0.0032483
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0032788, upper bound: 0.0032505
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032893
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 5, lower bound: -0.0032500, upper bound: 0.0032952

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040656, upper bound: 0.0038397
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040503, upper bound: 0.0038417
time: 1.48 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038690, upper bound: 0.0038862
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038694, upper bound: 0.0038855
time: 1.77 seconds

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040185, upper bound: 0.0038609
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040185, upper bound: 0.0038609
time: 1.84 seconds

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040803, upper bound: 0.0040131
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040727, upper bound: 0.0040294
time: 1.86 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037097, upper bound: 0.0037130
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037097, upper bound: 0.0037129
time: 1.83 seconds

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038311, upper bound: 0.0037955
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038163, upper bound: 0.0038370
time: 1.79 seconds

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
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041086, upper bound: 0.0040755
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041086, upper bound: 0.0040755
time: 2.01 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040634, upper bound: 0.0040507
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040540, upper bound: 0.0040599
time: 1.40 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0041015, upper bound: 0.0039689
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040958, upper bound: 0.0039714
time: 2.03 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040998, upper bound: 0.0040103
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040984, upper bound: 0.0040185
time: 1.41 seconds

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
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037171, upper bound: 0.0037517
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036712, upper bound: 0.0037901
time: 1.72 seconds

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
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037396, upper bound: 0.0037738
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0037100, upper bound: 0.0038162
time: 1.71 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040733, upper bound: 0.0040154
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040440, upper bound: 0.0040313
time: 1.45 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040880, upper bound: 0.0040560
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040869, upper bound: 0.0040568
time: 1.44 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039898, upper bound: 0.0038324
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039466, upper bound: 0.0038549
time: 1.77 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035705, upper bound: 0.0035726
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035705, upper bound: 0.0035726
time: 1.51 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036322, upper bound: 0.0035577
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036322, upper bound: 0.0035577
time: 1.46 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038297, upper bound: 0.0038530
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038290, upper bound: 0.0038591
time: 1.68 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033560, upper bound: 0.0033743
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033560, upper bound: 0.0033743
time: 1.00 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038180, upper bound: 0.0038506
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038180, upper bound: 0.0038507
time: 1.38 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035958, upper bound: 0.0036276
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035938, upper bound: 0.0036319
time: 1.21 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036145, upper bound: 0.0036165
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035867, upper bound: 0.0036514
time: 1.37 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033068, upper bound: 0.0032902
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033068, upper bound: 0.0032902
time: 1.16 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032635, upper bound: 0.0032510
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032450, upper bound: 0.0032850
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032632, upper bound: 0.0032629
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032632, upper bound: 0.0032629
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036301
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036298, upper bound: 0.0036404
time: 1.36 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035486, upper bound: 0.0035466
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035486, upper bound: 0.0035466
time: 1.50 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035808, upper bound: 0.0035791
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035808, upper bound: 0.0035790
time: 1.49 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039991, upper bound: 0.0039238
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039901, upper bound: 0.0039243
time: 1.79 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040255, upper bound: 0.0040352
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040167, upper bound: 0.0040530
time: 1.87 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040656, upper bound: 0.0040293
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040638, upper bound: 0.0040318
time: 1.78 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040680, upper bound: 0.0040585
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040677, upper bound: 0.0040600
time: 1.88 seconds

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
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035568, upper bound: 0.0035124
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035163, upper bound: 0.0035544
time: 1.45 seconds

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036071, upper bound: 0.0036039
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036070, upper bound: 0.0036043
time: 1.59 seconds

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035944, upper bound: 0.0035846
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035944, upper bound: 0.0035846
time: 1.59 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039961, upper bound: 0.0039879
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039858, upper bound: 0.0040045
time: 1.45 seconds

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040723, upper bound: 0.0040903
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040401, upper bound: 0.0041086
time: 2.02 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038734, upper bound: 0.0038934
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038734, upper bound: 0.0038934
time: 1.38 seconds

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039416, upper bound: 0.0040044
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039416, upper bound: 0.0040044
time: 1.87 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040568, upper bound: 0.0041457
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0040501, upper bound: 0.0041566
time: 1.91 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039331, upper bound: 0.0039179
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038787, upper bound: 0.0039749
time: 1.83 seconds

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038831, upper bound: 0.0039251
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038836, upper bound: 0.0039233
time: 1.83 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038907, upper bound: 0.0038546
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038907, upper bound: 0.0038546
time: 1.58 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039789, upper bound: 0.0041593
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039749, upper bound: 0.0041602
time: 1.94 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032472, upper bound: 0.0032444
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032472, upper bound: 0.0032444
time: 1.30 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036359, upper bound: 0.0036411
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0036366, upper bound: 0.0036405
time: 1.49 seconds

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035606, upper bound: 0.0035590
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035582, upper bound: 0.0035613
time: 1.60 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033503, upper bound: 0.0033508
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033503, upper bound: 0.0033509
time: 1.33 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038377, upper bound: 0.0038374
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038377, upper bound: 0.0038374
time: 1.41 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039005, upper bound: 0.0038679
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0038475, upper bound: 0.0039022
time: 1.82 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034242, upper bound: 0.0034229
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034242, upper bound: 0.0034229
time: 1.34 seconds

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039701, upper bound: 0.0039406
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0039329, upper bound: 0.0039804
time: 1.60 seconds

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
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032844, upper bound: 0.0032598
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032622
time: 1.22 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032954, upper bound: 0.0033044
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032912, upper bound: 0.0033079
time: 1.04 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032790, upper bound: 0.0032669
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032767, upper bound: 0.0032710
time: 1.20 seconds

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
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032796
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032663, upper bound: 0.0032999
time: 1.23 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032539, upper bound: 0.0032142
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032502, upper bound: 0.0032193
time: 1.22 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040656, upper bound: 0.0038397
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040503, upper bound: 0.0038417
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038690, upper bound: 0.0038862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038694, upper bound: 0.0038855
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040185, upper bound: 0.0038609
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040185, upper bound: 0.0038609
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040803, upper bound: 0.0040131
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040727, upper bound: 0.0040294
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0037097, upper bound: 0.0037130
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0037097, upper bound: 0.0037129
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038311, upper bound: 0.0037955
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038163, upper bound: 0.0038370
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0041086, upper bound: 0.0040755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0041086, upper bound: 0.0040755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040634, upper bound: 0.0040507
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040540, upper bound: 0.0040599
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0041015, upper bound: 0.0039689
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040958, upper bound: 0.0039714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040998, upper bound: 0.0040103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040984, upper bound: 0.0040185
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0037171, upper bound: 0.0037517
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036712, upper bound: 0.0037901
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0037396, upper bound: 0.0037738
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0037100, upper bound: 0.0038162
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040733, upper bound: 0.0040154
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040440, upper bound: 0.0040313
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040880, upper bound: 0.0040560
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040869, upper bound: 0.0040568
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039898, upper bound: 0.0038324
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039466, upper bound: 0.0038549
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035705, upper bound: 0.0035726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035705, upper bound: 0.0035726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036322, upper bound: 0.0035577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036322, upper bound: 0.0035577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038297, upper bound: 0.0038530
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038290, upper bound: 0.0038591
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0033560, upper bound: 0.0033743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0033560, upper bound: 0.0033743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038180, upper bound: 0.0038506
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038180, upper bound: 0.0038507
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035958, upper bound: 0.0036276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035938, upper bound: 0.0036319
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036145, upper bound: 0.0036165
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035867, upper bound: 0.0036514
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0033068, upper bound: 0.0032902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0033068, upper bound: 0.0032902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032635, upper bound: 0.0032510
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032450, upper bound: 0.0032850
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032632, upper bound: 0.0032629
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032632, upper bound: 0.0032629
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036400, upper bound: 0.0036301
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036298, upper bound: 0.0036404
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035486, upper bound: 0.0035466
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035486, upper bound: 0.0035466
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035808, upper bound: 0.0035791
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035808, upper bound: 0.0035790
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039991, upper bound: 0.0039238
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039901, upper bound: 0.0039243
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040255, upper bound: 0.0040352
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040167, upper bound: 0.0040530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040656, upper bound: 0.0040293
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040638, upper bound: 0.0040318
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040680, upper bound: 0.0040585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040677, upper bound: 0.0040600
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035568, upper bound: 0.0035124
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035163, upper bound: 0.0035544
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036071, upper bound: 0.0036039
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036070, upper bound: 0.0036043
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035944, upper bound: 0.0035846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035944, upper bound: 0.0035846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039961, upper bound: 0.0039879
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039858, upper bound: 0.0040045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040723, upper bound: 0.0040903
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040401, upper bound: 0.0041086
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038734, upper bound: 0.0038934
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038734, upper bound: 0.0038934
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039416, upper bound: 0.0040044
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039416, upper bound: 0.0040044
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040568, upper bound: 0.0041457
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0040501, upper bound: 0.0041566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039331, upper bound: 0.0039179
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038787, upper bound: 0.0039749
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038831, upper bound: 0.0039251
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038836, upper bound: 0.0039233
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038907, upper bound: 0.0038546
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038907, upper bound: 0.0038546
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039789, upper bound: 0.0041593
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039749, upper bound: 0.0041602
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032472, upper bound: 0.0032444
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032472, upper bound: 0.0032444
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036359, upper bound: 0.0036411
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0036366, upper bound: 0.0036405
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035606, upper bound: 0.0035590
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0035582, upper bound: 0.0035613
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0033503, upper bound: 0.0033508
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0033503, upper bound: 0.0033509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038377, upper bound: 0.0038374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038377, upper bound: 0.0038374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039005, upper bound: 0.0038679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0038475, upper bound: 0.0039022
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0034242, upper bound: 0.0034229
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0034242, upper bound: 0.0034229
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039701, upper bound: 0.0039406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0039329, upper bound: 0.0039804
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032844, upper bound: 0.0032598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032803, upper bound: 0.0032622
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032954, upper bound: 0.0033044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032912, upper bound: 0.0033079
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032790, upper bound: 0.0032669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032767, upper bound: 0.0032710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032823, upper bound: 0.0032796
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032663, upper bound: 0.0032999
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032539, upper bound: 0.0032142
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 5, lower bound: -0.0032502, upper bound: 0.0032193
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 5, lower bound: -0.0032788, upper bound: 0.0032505
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 5, lower bound: -0.0032557, upper bound: 0.0032893
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 5, lower bound: -0.0032500, upper bound: 0.0032952
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.006939222104847431
rel_dist={5: [-0.004486465180227661, 0.004485917759426528]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035579, upper bound: 0.0035317
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0035322, upper bound: 0.0035576
time: 1.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.92
Output dim: 5, lower bound: -0.0035579, upper bound: 0.0035317
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.92
Output dim: 5, lower bound: -0.0035322, upper bound: 0.0035576

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034752, upper bound: 0.0034570
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034827, upper bound: 0.0034532
time: 1.82 seconds

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
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034748, upper bound: 0.0034919
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034665, upper bound: 0.0034995
time: 1.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.82
Output dim: 5, lower bound: -0.0034752, upper bound: 0.0034570
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.82
Output dim: 5, lower bound: -0.0034827, upper bound: 0.0034532
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.82
Output dim: 5, lower bound: -0.0034748, upper bound: 0.0034919
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.82
Output dim: 5, lower bound: -0.0034665, upper bound: 0.0034995

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160218, 0.0160240
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045172, 0.0045178
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0333287, 0.0333331
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044105, 0.0044111
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249111, 0.0249078
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069210, 0.0069201
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062822, 0.0062814
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234441, 0.0234410
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182442, 0.0182466
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015742, 0.0015740

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034290, upper bound: 0.0034151
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034289, upper bound: 0.0034127
time: 1.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160049, 0.0160442
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045124, 0.0045235
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0332935, 0.0333752
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044059, 0.0044167
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249426, 0.0248815
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069298, 0.0069128
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062902, 0.0062748
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234737, 0.0234163
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182249, 0.0182696
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015762, 0.0015724

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034588, upper bound: 0.0034275
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034303
time: 1.66 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034307, upper bound: 0.0034449
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034339, upper bound: 0.0034452
time: 1.83 seconds

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034618, upper bound: 0.0034633
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034353, upper bound: 0.0034949
time: 1.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.78 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034290, upper bound: 0.0034151
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034289, upper bound: 0.0034127
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034588, upper bound: 0.0034275
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034518, upper bound: 0.0034303
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034307, upper bound: 0.0034449
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034339, upper bound: 0.0034452
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034618, upper bound: 0.0034633
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.78
Output dim: 5, lower bound: -0.0034353, upper bound: 0.0034949

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157745, 0.0157668
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044474, 0.0044452
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328142, 0.0327981
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043424, 0.0043403
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245112, 0.0245233
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068099, 0.0068133
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061814, 0.0061844
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230678, 0.0230792
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179626, 0.0179537
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015490, 0.0015497

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034135, upper bound: 0.0033992
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034106, upper bound: 0.0033999
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157646, 0.0157731
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044446, 0.0044470
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327937, 0.0328112
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043397, 0.0043420
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245211, 0.0245079
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068127, 0.0068090
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061839, 0.0061805
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230771, 0.0230647
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179513, 0.0179609
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015496, 0.0015488

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034122, upper bound: 0.0033961
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034122, upper bound: 0.0033962
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158212, 0.0158698
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044606, 0.0044743
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0329113, 0.0330124
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043553, 0.0043687
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246714, 0.0245959
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068544, 0.0068335
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062218, 0.0062027
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232185, 0.0231475
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180157, 0.0180710
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015591, 0.0015543

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034129, upper bound: 0.0033861
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034128, upper bound: 0.0033835
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158305, 0.0158550
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044632, 0.0044701
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0329307, 0.0329817
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043579, 0.0043646
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246484, 0.0246104
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068481, 0.0068375
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062160, 0.0062064
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231969, 0.0231611
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180263, 0.0180542
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015576, 0.0015552

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033816, upper bound: 0.0033554
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033764, upper bound: 0.0033623
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158643, 0.0158461
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044727, 0.0044676
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0330009, 0.0329631
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043671, 0.0043621
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246346, 0.0246628
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068442, 0.0068521
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062125, 0.0062196
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231839, 0.0232105
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180647, 0.0180441
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015568, 0.0015585

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034135, upper bound: 0.0034287
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034119, upper bound: 0.0034286
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158579, 0.0158620
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044709, 0.0044721
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0329877, 0.0329961
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043654, 0.0043665
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246592, 0.0246530
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068511, 0.0068493
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062187, 0.0062171
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232071, 0.0232012
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180575, 0.0180621
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015583, 0.0015579

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033432, upper bound: 0.0033603
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033460, upper bound: 0.0033538
time: 1.58 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034391, upper bound: 0.0034381
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034359, upper bound: 0.0034407
time: 1.96 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034280, upper bound: 0.0034676
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034122, upper bound: 0.0034877
time: 1.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034135, upper bound: 0.0033992
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034106, upper bound: 0.0033999
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034122, upper bound: 0.0033961
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034122, upper bound: 0.0033962
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034129, upper bound: 0.0033861
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034128, upper bound: 0.0033835
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0033816, upper bound: 0.0033554
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0033764, upper bound: 0.0033623
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034135, upper bound: 0.0034287
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034119, upper bound: 0.0034286
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0033432, upper bound: 0.0033603
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0033460, upper bound: 0.0033538
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034391, upper bound: 0.0034381
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034359, upper bound: 0.0034407
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034280, upper bound: 0.0034676
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0034122, upper bound: 0.0034877

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157552, 0.0157500
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044420, 0.0044405
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327739, 0.0327632
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043371, 0.0043357
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244852, 0.0244932
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068027, 0.0068049
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061748, 0.0061768
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230433, 0.0230508
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179405, 0.0179346
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015473, 0.0015478

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033175, upper bound: 0.0033003
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033157, upper bound: 0.0033013
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157563, 0.0157474
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044423, 0.0044398
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327762, 0.0327578
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043374, 0.0043350
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244811, 0.0244949
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068016, 0.0068054
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061738, 0.0061773
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230394, 0.0230524
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179418, 0.0179316
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015471, 0.0015479

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034034, upper bound: 0.0033693
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033851, upper bound: 0.0033930
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156932, 0.0157023
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044245, 0.0044271
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326450, 0.0326640
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043200, 0.0043226
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244110, 0.0243968
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067821, 0.0067782
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061561, 0.0061525
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229735, 0.0229601
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178699, 0.0178803
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015426, 0.0015417

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033868, upper bound: 0.0033674
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033838, upper bound: 0.0033702
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156939, 0.0156993
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044247, 0.0044262
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326464, 0.0326576
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043202, 0.0043217
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244063, 0.0243979
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067808, 0.0067785
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061549, 0.0061528
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229690, 0.0229611
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178707, 0.0178768
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015423, 0.0015418

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033478, upper bound: 0.0033247
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033414, upper bound: 0.0033312
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156209, 0.0156502
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044041, 0.0044124
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0324946, 0.0325556
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043001, 0.0043082
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243300, 0.0242844
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067596, 0.0067469
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061357, 0.0061242
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228972, 0.0228543
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177876, 0.0178210
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015375, 0.0015346

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034002, upper bound: 0.0033637
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033890, upper bound: 0.0033728
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156016, 0.0156509
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043987, 0.0044126
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0324545, 0.0325571
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042948, 0.0043084
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243312, 0.0242545
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067599, 0.0067386
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061360, 0.0061166
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228983, 0.0228262
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177656, 0.0178218
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015376, 0.0015327

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034029, upper bound: 0.0033695
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034018, upper bound: 0.0033732
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157888, 0.0158199
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044515, 0.0044602
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328440, 0.0329087
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043464, 0.0043549
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245939, 0.0245456
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068329, 0.0068195
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062022, 0.0061900
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231456, 0.0231001
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179789, 0.0180143
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015542, 0.0015511

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033770, upper bound: 0.0033251
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033521, upper bound: 0.0033508
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157960, 0.0158133
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044535, 0.0044584
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328590, 0.0328950
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043484, 0.0043531
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245836, 0.0245567
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068301, 0.0068226
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061996, 0.0061929
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231359, 0.0231106
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179870, 0.0180067
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015535, 0.0015518

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033631, upper bound: 0.0033492
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033641, upper bound: 0.0033493
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157817, 0.0157718
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044494, 0.0044467
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328290, 0.0328087
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043444, 0.0043417
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245191, 0.0245344
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068121, 0.0068164
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061834, 0.0061872
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230752, 0.0230896
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179707, 0.0179595
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015495, 0.0015504

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033943, upper bound: 0.0034090
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033943, upper bound: 0.0034084
time: 2.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157889, 0.0157635
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044515, 0.0044443
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328441, 0.0327913
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043464, 0.0043394
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245062, 0.0245457
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068085, 0.0068195
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061801, 0.0061901
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230630, 0.0231002
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179789, 0.0179500
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015486, 0.0015511

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033967, upper bound: 0.0034132
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033963, upper bound: 0.0034137
time: 2.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157463, 0.0157301
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044395, 0.0044349
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327555, 0.0327217
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043347, 0.0043302
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244542, 0.0244794
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067941, 0.0068011
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061670, 0.0061733
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230141, 0.0230378
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179304, 0.0179119
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015454, 0.0015469

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033185, upper bound: 0.0033214
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033016, upper bound: 0.0033355
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157260, 0.0157405
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044338, 0.0044378
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327133, 0.0327435
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043291, 0.0043331
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244704, 0.0244479
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067986, 0.0067923
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061711, 0.0061654
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230294, 0.0230082
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179073, 0.0179238
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015464, 0.0015450

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033241, upper bound: 0.0033251
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033150, upper bound: 0.0033311
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160582, 0.0160661
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045274, 0.0045296
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334044, 0.0334207
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044205, 0.0044227
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249765, 0.0249644
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069392, 0.0069358
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062987, 0.0062957
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235057, 0.0234943
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182856, 0.0182945
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015784, 0.0015776

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033627, upper bound: 0.0033609
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033617, upper bound: 0.0033609
time: 1.45 seconds

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0028934, upper bound: 0.0028811
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0028934, upper bound: 0.0028811
time: 1.05 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0034075, upper bound: 0.0034313
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033983, upper bound: 0.0034464
time: 1.71 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033440, upper bound: 0.0034124
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033330, upper bound: 0.0034208
time: 1.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033175, upper bound: 0.0033003
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033157, upper bound: 0.0033013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0034034, upper bound: 0.0033693
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033851, upper bound: 0.0033930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033868, upper bound: 0.0033674
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033838, upper bound: 0.0033702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033478, upper bound: 0.0033247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033414, upper bound: 0.0033312
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0034002, upper bound: 0.0033637
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033890, upper bound: 0.0033728
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0034029, upper bound: 0.0033695
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0034018, upper bound: 0.0033732
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033770, upper bound: 0.0033251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033521, upper bound: 0.0033508
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033631, upper bound: 0.0033492
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033641, upper bound: 0.0033493
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033943, upper bound: 0.0034090
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033943, upper bound: 0.0034084
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033967, upper bound: 0.0034132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033963, upper bound: 0.0034137
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033185, upper bound: 0.0033214
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033016, upper bound: 0.0033355
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033241, upper bound: 0.0033251
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033150, upper bound: 0.0033311
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033627, upper bound: 0.0033609
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033617, upper bound: 0.0033609
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0028934, upper bound: 0.0028811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0028934, upper bound: 0.0028811
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0034075, upper bound: 0.0034313
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033983, upper bound: 0.0034464
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033440, upper bound: 0.0034124
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.27
Output dim: 5, lower bound: -0.0033330, upper bound: 0.0034208

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156807, 0.0156977
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044210, 0.0044258
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326191, 0.0326545
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043166, 0.0043213
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244039, 0.0243775
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067801, 0.0067728
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061543, 0.0061476
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229668, 0.0229419
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178557, 0.0178751
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015422, 0.0015405

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032545, upper bound: 0.0032350
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032545, upper bound: 0.0032350
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157028, 0.0156756
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044272, 0.0044195
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326649, 0.0326084
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043227, 0.0043152
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243695, 0.0244117
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067706, 0.0067823
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061456, 0.0061563
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229344, 0.0229742
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178808, 0.0178499
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015400, 0.0015427

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032901, upper bound: 0.0032599
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032749, upper bound: 0.0032766
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157129, 0.0157157
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044300, 0.0044308
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326859, 0.0326918
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043255, 0.0043262
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244318, 0.0244274
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067879, 0.0067867
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061613, 0.0061602
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229931, 0.0229889
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178923, 0.0178955
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015439, 0.0015437

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033250, upper bound: 0.0032761
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033178, upper bound: 0.0032864
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157243, 0.0157040
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044333, 0.0044275
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327098, 0.0326674
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043286, 0.0043230
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244136, 0.0244452
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067828, 0.0067916
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061568, 0.0061647
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229759, 0.0230057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179054, 0.0178822
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015428, 0.0015448

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033653, upper bound: 0.0033720
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033626, upper bound: 0.0033725
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154342, 0.0154599
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043515, 0.0043587
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321062, 0.0321597
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042487, 0.0042558
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0240341, 0.0239942
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066774, 0.0066663
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060611, 0.0060510
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226188, 0.0225812
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175750, 0.0176043
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015188, 0.0015163

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033681, upper bound: 0.0033103
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033273, upper bound: 0.0033491
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154507, 0.0154364
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043561, 0.0043521
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321407, 0.0321109
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042533, 0.0042494
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0239977, 0.0240200
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066673, 0.0066735
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060519, 0.0060575
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0225845, 0.0226054
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175939, 0.0175775
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015165, 0.0015179

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032938, upper bound: 0.0032743
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032867, upper bound: 0.0032777
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156531, 0.0156655
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044132, 0.0044167
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325617, 0.0325873
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043090, 0.0043124
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243537, 0.0243346
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067662, 0.0067609
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061417, 0.0061368
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229196, 0.0229015
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178243, 0.0178383
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015390, 0.0015378

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033434, upper bound: 0.0032924
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033135, upper bound: 0.0033203
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156596, 0.0156585
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044150, 0.0044147
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325752, 0.0325729
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043108, 0.0043105
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243430, 0.0243447
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067632, 0.0067637
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061389, 0.0061394
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229094, 0.0229110
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178317, 0.0178304
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015383, 0.0015384

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030826, upper bound: 0.0030631
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030826, upper bound: 0.0030631
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0153652, 0.0154317
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043320, 0.0043508
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0319628, 0.0321012
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042298, 0.0042481
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0239904, 0.0238870
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066652, 0.0066365
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060500, 0.0060240
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0225776, 0.0224803
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0174965, 0.0175722
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015160, 0.0015095

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032158, upper bound: 0.0031940
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032158, upper bound: 0.0031940
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154030, 0.0153945
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043427, 0.0043403
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0320413, 0.0320238
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042402, 0.0042378
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0239326, 0.0239457
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066492, 0.0066528
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060354, 0.0060387
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0225232, 0.0225355
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175394, 0.0175299
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015124, 0.0015132

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033225, upper bound: 0.0032762
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033012, upper bound: 0.0033066
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155153, 0.0155645
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043744, 0.0043882
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322750, 0.0323774
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042711, 0.0042846
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241968, 0.0241204
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067226, 0.0067013
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061021, 0.0060828
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227719, 0.0226999
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176674, 0.0177234
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015291, 0.0015243

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033806, upper bound: 0.0033358
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033736, upper bound: 0.0033491
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155152, 0.0155662
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043743, 0.0043887
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322747, 0.0323808
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042710, 0.0042851
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241994, 0.0241201
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067233, 0.0067013
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061027, 0.0060827
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227743, 0.0226997
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176672, 0.0177253
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015293, 0.0015242

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033840, upper bound: 0.0033225
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033380, upper bound: 0.0033551
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159187, 0.0160137
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044881, 0.0045149
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0331141, 0.0333117
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043821, 0.0044083
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0248951, 0.0247474
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069166, 0.0068756
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062782, 0.0062409
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234290, 0.0232901
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0181267, 0.0182349
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015732, 0.0015639

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033006, upper bound: 0.0032209
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032801, upper bound: 0.0032489
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159842, 0.0159498
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045065, 0.0044968
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0332504, 0.0331787
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044002, 0.0043907
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0247957, 0.0248493
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068890, 0.0069039
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062531, 0.0062666
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0233355, 0.0233860
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182013, 0.0181621
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015669, 0.0015703

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032785, upper bound: 0.0032649
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032744, upper bound: 0.0032764
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157255, 0.0157466
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044336, 0.0044395
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327123, 0.0327561
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043289, 0.0043347
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244798, 0.0244471
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068012, 0.0067921
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061735, 0.0061652
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230382, 0.0230075
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179067, 0.0179307
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015470, 0.0015449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031372, upper bound: 0.0031316
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031372, upper bound: 0.0031316
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157293, 0.0157462
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044347, 0.0044394
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327201, 0.0327553
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043300, 0.0043346
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244793, 0.0244529
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068011, 0.0067937
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061733, 0.0061667
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230377, 0.0230129
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179110, 0.0179303
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015469, 0.0015453

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033454, upper bound: 0.0032961
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033053, upper bound: 0.0033304
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157069, 0.0157001
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044283, 0.0044264
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326735, 0.0326594
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043238, 0.0043220
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244076, 0.0244181
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067812, 0.0067841
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061552, 0.0061579
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229703, 0.0229802
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178855, 0.0178778
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015424, 0.0015431

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033736, upper bound: 0.0033824
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033609, upper bound: 0.0033852
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157099, 0.0156964
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044292, 0.0044254
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326798, 0.0326517
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043247, 0.0043209
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244018, 0.0244228
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067796, 0.0067854
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061538, 0.0061591
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229648, 0.0229846
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178890, 0.0178736
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015420, 0.0015434

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033864, upper bound: 0.0033816
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033651, upper bound: 0.0034008
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157700, 0.0157468
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044462, 0.0044396
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328048, 0.0327565
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043412, 0.0043348
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244802, 0.0245163
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068013, 0.0068113
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061735, 0.0061826
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230386, 0.0230725
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179574, 0.0179310
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015470, 0.0015493

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033026, upper bound: 0.0033223
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033045, upper bound: 0.0033164
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157717, 0.0157446
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044466, 0.0044390
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328083, 0.0327520
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043417, 0.0043342
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244768, 0.0245189
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068004, 0.0068121
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061727, 0.0061833
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230354, 0.0230750
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179593, 0.0179285
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015468, 0.0015494

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031248, upper bound: 0.0031477
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031248, upper bound: 0.0031477
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156121, 0.0156174
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044016, 0.0044031
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0324764, 0.0324874
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042977, 0.0042992
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242791, 0.0242708
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067454, 0.0067432
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061228, 0.0061207
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228493, 0.0228416
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177776, 0.0177836
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015343, 0.0015338

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032982, upper bound: 0.0032951
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032849, upper bound: 0.0032983
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156282, 0.0155959
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044062, 0.0043971
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325098, 0.0324427
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043022, 0.0042933
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242456, 0.0242958
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067361, 0.0067501
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061144, 0.0061270
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228178, 0.0228650
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177959, 0.0177591
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015322, 0.0015353

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030580, upper bound: 0.0030919
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030580, upper bound: 0.0030919
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156345, 0.0156688
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044080, 0.0044176
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325230, 0.0325944
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043039, 0.0043133
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243590, 0.0243056
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067676, 0.0067528
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061430, 0.0061295
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229245, 0.0228743
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178031, 0.0178422
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015393, 0.0015360

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033021, upper bound: 0.0032971
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032991, upper bound: 0.0033026
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156580, 0.0156490
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044146, 0.0044120
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325717, 0.0325531
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043103, 0.0043079
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243282, 0.0243421
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067591, 0.0067630
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061352, 0.0061387
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228955, 0.0229086
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178298, 0.0178196
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015374, 0.0015383

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032331, upper bound: 0.0032343
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032083, upper bound: 0.0032557
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159851, 0.0160639
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045068, 0.0045290
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0332522, 0.0334163
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044004, 0.0044221
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249732, 0.0248506
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069383, 0.0069042
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062979, 0.0062670
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0235026, 0.0233872
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182023, 0.0182921
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015782, 0.0015704

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032604, upper bound: 0.0032539
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032604, upper bound: 0.0032543
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160059, 0.0160374
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045127, 0.0045215
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0332956, 0.0333610
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044061, 0.0044148
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249320, 0.0248831
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069268, 0.0069133
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062875, 0.0062751
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234638, 0.0234177
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182261, 0.0182619
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015755, 0.0015725

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033545, upper bound: 0.0033349
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033347, upper bound: 0.0033537
time: 1.64 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033520, upper bound: 0.0033592
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033394, upper bound: 0.0033785
time: 2.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160423
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045229
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0333712
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044162
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249396, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069290, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062894, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234709, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182675
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015760, 0.0015784

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033270, upper bound: 0.0033682
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033140, upper bound: 0.0033746
time: 2.24 seconds

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033842
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033188, upper bound: 0.0033871
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160661, 0.0160205
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045296, 0.0045168
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0334207, 0.0333258
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044227, 0.0044101
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0249056, 0.0249765
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069195, 0.0069392
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062808, 0.0062987
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234390, 0.0235057
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182945, 0.0182426
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015739, 0.0015784

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031550, upper bound: 0.0032252
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031550, upper bound: 0.0032253
time: 1.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032545, upper bound: 0.0032350
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032545, upper bound: 0.0032350
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032901, upper bound: 0.0032599
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032749, upper bound: 0.0032766
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033250, upper bound: 0.0032761
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033178, upper bound: 0.0032864
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033653, upper bound: 0.0033720
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033626, upper bound: 0.0033725
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033681, upper bound: 0.0033103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033273, upper bound: 0.0033491
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032938, upper bound: 0.0032743
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032867, upper bound: 0.0032777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033434, upper bound: 0.0032924
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033135, upper bound: 0.0033203
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0030826, upper bound: 0.0030631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0030826, upper bound: 0.0030631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032158, upper bound: 0.0031940
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032158, upper bound: 0.0031940
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033225, upper bound: 0.0032762
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033012, upper bound: 0.0033066
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033806, upper bound: 0.0033358
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033736, upper bound: 0.0033491
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033840, upper bound: 0.0033225
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033380, upper bound: 0.0033551
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033006, upper bound: 0.0032209
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032801, upper bound: 0.0032489
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032785, upper bound: 0.0032649
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032744, upper bound: 0.0032764
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0031372, upper bound: 0.0031316
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0031372, upper bound: 0.0031316
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033454, upper bound: 0.0032961
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033053, upper bound: 0.0033304
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033736, upper bound: 0.0033824
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033609, upper bound: 0.0033852
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033864, upper bound: 0.0033816
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033651, upper bound: 0.0034008
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033026, upper bound: 0.0033223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033045, upper bound: 0.0033164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0031248, upper bound: 0.0031477
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0031248, upper bound: 0.0031477
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032982, upper bound: 0.0032951
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032849, upper bound: 0.0032983
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0030580, upper bound: 0.0030919
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0030580, upper bound: 0.0030919
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033021, upper bound: 0.0032971
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032991, upper bound: 0.0033026
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032331, upper bound: 0.0032343
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032083, upper bound: 0.0032557
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032604, upper bound: 0.0032539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0032604, upper bound: 0.0032543
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033545, upper bound: 0.0033349
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033347, upper bound: 0.0033537
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033520, upper bound: 0.0033592
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033394, upper bound: 0.0033785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033270, upper bound: 0.0033682
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033140, upper bound: 0.0033746
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033192, upper bound: 0.0033842
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0033188, upper bound: 0.0033871
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0031550, upper bound: 0.0032252
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.70
Output dim: 5, lower bound: -0.0031550, upper bound: 0.0032253

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155659, 0.0155565
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043886, 0.0043860
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0323802, 0.0323607
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042850, 0.0042824
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241844, 0.0241989
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067191, 0.0067232
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060990, 0.0061026
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227602, 0.0227739
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177250, 0.0177143
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015283, 0.0015292

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031738, upper bound: 0.0031256
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031585, upper bound: 0.0031403
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155876, 0.0155387
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043947, 0.0043809
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0324254, 0.0323236
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042910, 0.0042775
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241567, 0.0242327
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067114, 0.0067326
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060920, 0.0061111
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227341, 0.0228056
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177497, 0.0176940
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015266, 0.0015314

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032488, upper bound: 0.0032373
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032385, upper bound: 0.0032512
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155483, 0.0155959
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043836, 0.0043971
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0323436, 0.0324426
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042802, 0.0042933
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242456, 0.0241716
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067361, 0.0067156
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061144, 0.0060957
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228178, 0.0227482
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177049, 0.0177591
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015322, 0.0015275

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032989, upper bound: 0.0032045
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032511, upper bound: 0.0032502
time: 1.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156073, 0.0155511
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044003, 0.0043844
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0324665, 0.0323495
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042964, 0.0042809
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241760, 0.0242634
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067168, 0.0067411
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060968, 0.0061189
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227523, 0.0228346
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177722, 0.0177082
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015278, 0.0015333

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032610, upper bound: 0.0032037
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032410, upper bound: 0.0032267
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157603, 0.0157298
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044434, 0.0044348
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327846, 0.0327213
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043385, 0.0043301
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244538, 0.0245011
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067940, 0.0068071
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061669, 0.0061788
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230138, 0.0230583
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179463, 0.0179117
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015453, 0.0015483

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032266, upper bound: 0.0032340
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032266, upper bound: 0.0032339
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157502, 0.0157395
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044406, 0.0044376
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327636, 0.0327415
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043357, 0.0043328
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244689, 0.0244855
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067982, 0.0068028
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061707, 0.0061749
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230280, 0.0230435
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179348, 0.0179227
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015463, 0.0015473

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032639, upper bound: 0.0032740
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032639, upper bound: 0.0032740
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154439, 0.0154713
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043542, 0.0043619
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321264, 0.0321834
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042514, 0.0042590
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0240519, 0.0240093
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066823, 0.0066705
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060655, 0.0060548
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226355, 0.0225954
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175860, 0.0176173
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015199, 0.0015172

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033581, upper bound: 0.0032977
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033568, upper bound: 0.0033004
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154456, 0.0154700
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043547, 0.0043616
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321300, 0.0321808
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042519, 0.0042586
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0240499, 0.0240119
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066818, 0.0066712
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060650, 0.0060555
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226336, 0.0225979
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175880, 0.0176158
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015198, 0.0015174

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031284, upper bound: 0.0031449
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031303, upper bound: 0.0031444
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0153758, 0.0153706
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043350, 0.0043336
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0319847, 0.0319740
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042327, 0.0042313
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0238954, 0.0239034
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066388, 0.0066411
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060261, 0.0060281
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0224882, 0.0224957
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175085, 0.0175026
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015100, 0.0015105

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0029540, upper bound: 0.0029424
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0029540, upper bound: 0.0029424
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0153852, 0.0153614
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043377, 0.0043310
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0320044, 0.0319549
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042353, 0.0042287
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0238811, 0.0239181
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066349, 0.0066452
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060225, 0.0060318
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0224748, 0.0225096
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175193, 0.0174921
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015091, 0.0015115

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032631, upper bound: 0.0032519
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032586, upper bound: 0.0032550
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158093, 0.0158895
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044572, 0.0044798
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0328865, 0.0330533
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043520, 0.0043741
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0247020, 0.0245773
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068629, 0.0068283
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062295, 0.0061980
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232473, 0.0231300
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180021, 0.0180934
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015610, 0.0015531

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033224, upper bound: 0.0032648
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033106, upper bound: 0.0032695
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158741, 0.0158216
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044755, 0.0044607
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0330214, 0.0329122
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043699, 0.0043554
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245965, 0.0246781
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068336, 0.0068563
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062029, 0.0062235
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231481, 0.0232249
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180759, 0.0180162
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015543, 0.0015595

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032598, upper bound: 0.0032531
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032438, upper bound: 0.0032625
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0152280, 0.0153009
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0042933, 0.0043139
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0316773, 0.0318290
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0041920, 0.0042121
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0237870, 0.0236736
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066087, 0.0065772
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0059987, 0.0059701
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0223862, 0.0222795
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0173402, 0.0174232
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015032, 0.0014960

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032966, upper bound: 0.0032475
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032953, upper bound: 0.0032499
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0153157, 0.0152196
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043181, 0.0042910
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0318597, 0.0316598
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042161, 0.0041897
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0236605, 0.0238099
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0065736, 0.0066151
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0059668, 0.0060045
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0222672, 0.0224078
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0174400, 0.0173306
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0014952, 0.0015046

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030286, upper bound: 0.0030114
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030286, upper bound: 0.0030114
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154205, 0.0154830
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043476, 0.0043652
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0320777, 0.0322079
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042450, 0.0042622
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0240702, 0.0239729
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066874, 0.0066604
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060701, 0.0060456
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226527, 0.0225612
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175594, 0.0176306
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015211, 0.0015149

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033600, upper bound: 0.0033138
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033595, upper bound: 0.0033163
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154414, 0.0154697
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043535, 0.0043615
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321212, 0.0321801
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042507, 0.0042585
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0240494, 0.0240054
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066816, 0.0066694
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060649, 0.0060538
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226331, 0.0225918
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175832, 0.0176154
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015198, 0.0015170

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033553, upper bound: 0.0033314
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033568, upper bound: 0.0033313
time: 2.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155236, 0.0155759
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043767, 0.0043914
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322923, 0.0324010
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042734, 0.0042878
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242145, 0.0241332
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067275, 0.0067049
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061065, 0.0060860
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227885, 0.0227121
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176768, 0.0177363
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015302, 0.0015251

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033631, upper bound: 0.0032937
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033511, upper bound: 0.0033016
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155249, 0.0155745
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043770, 0.0043910
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322949, 0.0323981
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042737, 0.0042874
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242123, 0.0241352
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067269, 0.0067055
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061060, 0.0060866
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227865, 0.0227139
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176783, 0.0177348
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015301, 0.0015252

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033201, upper bound: 0.0033369
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033201, upper bound: 0.0033370
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157409, 0.0159141
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044379, 0.0044868
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327442, 0.0331047
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043332, 0.0043809
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0247404, 0.0244710
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068736, 0.0067988
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062392, 0.0061712
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232834, 0.0230299
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179242, 0.0181215
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015634, 0.0015464

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032773, upper bound: 0.0031924
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0031964
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158257, 0.0158359
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044618, 0.0044647
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0329206, 0.0329418
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043565, 0.0043593
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246187, 0.0246028
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068398, 0.0068354
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062085, 0.0062045
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231689, 0.0231540
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180208, 0.0180324
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015557, 0.0015547

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032365
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032646, upper bound: 0.0032363
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158175, 0.0158295
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044595, 0.0044629
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0329036, 0.0329285
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043543, 0.0043576
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246087, 0.0245901
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068370, 0.0068318
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062060, 0.0062013
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231595, 0.0231420
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180115, 0.0180251
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015551, 0.0015539

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0028768, upper bound: 0.0028754
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0028768, upper bound: 0.0028754
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158738, 0.0157830
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044754, 0.0044498
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0330207, 0.0328319
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043698, 0.0043448
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245365, 0.0246776
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068170, 0.0068562
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061877, 0.0062233
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230916, 0.0232243
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180755, 0.0179722
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015506, 0.0015595

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032497, upper bound: 0.0032415
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032384, upper bound: 0.0032506
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157365, 0.0157550
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044367, 0.0044419
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327352, 0.0327736
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043320, 0.0043371
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244929, 0.0244643
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068049, 0.0067969
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061768, 0.0061695
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230506, 0.0230236
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179193, 0.0179403
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015478, 0.0015460

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033236, upper bound: 0.0032676
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033155, upper bound: 0.0032737
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157380, 0.0157533
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044371, 0.0044415
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327383, 0.0327702
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043324, 0.0043366
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0244904, 0.0244666
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068042, 0.0067975
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061761, 0.0061701
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0230482, 0.0230258
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179210, 0.0179384
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015476, 0.0015461

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032197, upper bound: 0.0032186
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032050, upper bound: 0.0032436
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156175, 0.0156358
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044032, 0.0044083
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0324876, 0.0325257
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042992, 0.0043043
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243076, 0.0242792
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067534, 0.0067455
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061300, 0.0061229
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228762, 0.0228494
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0177837, 0.0178046
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015361, 0.0015343

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033696, upper bound: 0.0033539
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033383, upper bound: 0.0033783
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156333, 0.0156107
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044076, 0.0044012
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325205, 0.0324735
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043036, 0.0042974
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242687, 0.0243038
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067426, 0.0067523
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061202, 0.0061291
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228395, 0.0228726
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178018, 0.0177760
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015336, 0.0015358

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033366, upper bound: 0.0033074
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032809, upper bound: 0.0033590
time: 2.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156697, 0.0156682
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044179, 0.0044174
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325962, 0.0325929
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043136, 0.0043132
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243579, 0.0243604
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067674, 0.0067680
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061427, 0.0061433
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229235, 0.0229258
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178432, 0.0178414
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015393, 0.0015394

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033272, upper bound: 0.0033132
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033197, upper bound: 0.0033239
time: 2.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156818, 0.0156562
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044213, 0.0044141
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326214, 0.0325681
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043169, 0.0043099
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0243393, 0.0243792
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067622, 0.0067733
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061380, 0.0061481
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0229060, 0.0229435
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178570, 0.0178278
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015381, 0.0015406

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031560, upper bound: 0.0032026
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031614, upper bound: 0.0031978
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156521, 0.0156143
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044129, 0.0044023
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325595, 0.0324809
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043087, 0.0042983
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242742, 0.0243329
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067441, 0.0067604
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061216, 0.0061364
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228447, 0.0229000
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178231, 0.0177801
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015340, 0.0015377

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030527, upper bound: 0.0030651
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0030527, upper bound: 0.0030651
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156375, 0.0156306
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044088, 0.0044069
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325292, 0.0325149
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043047, 0.0043028
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0242996, 0.0243103
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067511, 0.0067541
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0061280, 0.0061307
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0228686, 0.0228787
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178065, 0.0177987
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015356, 0.0015363

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032798, upper bound: 0.0032763
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032660, upper bound: 0.0032925
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155227, 0.0155497
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043764, 0.0043840
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322903, 0.0323465
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042731, 0.0042805
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241738, 0.0241318
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067162, 0.0067045
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060963, 0.0060857
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227502, 0.0227107
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176758, 0.0177065
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015276, 0.0015250

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032149, upper bound: 0.0031913
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031910, upper bound: 0.0032136
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155392, 0.0155280
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043811, 0.0043779
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0323248, 0.0323013
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042777, 0.0042746
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241400, 0.0241575
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067068, 0.0067117
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060878, 0.0060922
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227184, 0.0227349
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176946, 0.0176818
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015255, 0.0015266

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032231, upper bound: 0.0032262
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032191, upper bound: 0.0032466
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154878, 0.0155348
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043666, 0.0043799
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322178, 0.0323156
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042635, 0.0042765
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241507, 0.0240776
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067098, 0.0066895
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060905, 0.0060720
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227285, 0.0226597
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176361, 0.0176896
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015262, 0.0015216

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032779, upper bound: 0.0032592
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032616, upper bound: 0.0032731
time: 2.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155005, 0.0155249
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043702, 0.0043770
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322442, 0.0322949
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042670, 0.0042737
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241352, 0.0240973
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067055, 0.0066950
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060865, 0.0060770
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227139, 0.0226783
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176505, 0.0176782
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015252, 0.0015228

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032946, upper bound: 0.0032762
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032673, upper bound: 0.0032977
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159696, 0.0160132
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045024, 0.0045147
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0332200, 0.0333108
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043961, 0.0044082
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0248944, 0.0248265
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069164, 0.0068975
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062780, 0.0062609
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234284, 0.0233645
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0181847, 0.0182344
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015732, 0.0015689

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033396, upper bound: 0.0033131
time: 13.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033336, upper bound: 0.0033196
time: 1.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159822, 0.0160010
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045060, 0.0045113
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0332462, 0.0332854
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043996, 0.0044048
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0248754, 0.0248462
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0069111, 0.0069030
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062732, 0.0062658
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0234105, 0.0233830
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0181990, 0.0182205
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015720, 0.0015701

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033129, upper bound: 0.0033278
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033051, upper bound: 0.0033298
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159085, 0.0158862
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044852, 0.0044789
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0330929, 0.0330464
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043793, 0.0043732
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246968, 0.0247316
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068615, 0.0068712
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062282, 0.0062369
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232425, 0.0232752
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0181151, 0.0180897
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015607, 0.0015629

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031971, upper bound: 0.0032068
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031971, upper bound: 0.0032068
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159301, 0.0158691
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044913, 0.0044741
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0331378, 0.0330110
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043853, 0.0043685
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246704, 0.0247651
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068542, 0.0068805
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062215, 0.0062454
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232176, 0.0233067
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0181397, 0.0180703
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015590, 0.0015650

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0028019, upper bound: 0.0028041
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0028019, upper bound: 0.0028041
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0159968, 0.0159247
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045101, 0.0044898
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0332765, 0.0331267
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044036, 0.0043838
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0247568, 0.0248688
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068782, 0.0069093
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062433, 0.0062715
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232989, 0.0234043
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182156, 0.0181336
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015645, 0.0015716

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033123, upper bound: 0.0033492
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033126, upper bound: 0.0033529
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0160440, 0.0158723
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0045234, 0.0044750
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0333749, 0.0330176
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0044166, 0.0043694
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246753, 0.0249423
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068555, 0.0069297
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062227, 0.0062901
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232222, 0.0234735
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0182695, 0.0180739
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015593, 0.0015762

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032935, upper bound: 0.0033520
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032923, upper bound: 0.0033550
time: 1.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158759, 0.0158435
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044760, 0.0044669
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0330250, 0.0329577
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043703, 0.0043614
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246305, 0.0246809
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068431, 0.0068571
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062115, 0.0062242
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231801, 0.0232274
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180780, 0.0180411
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015565, 0.0015597

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032600, upper bound: 0.0033074
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032495, upper bound: 0.0033312
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0158947, 0.0158256
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044813, 0.0044618
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0330643, 0.0329204
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043755, 0.0043565
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246027, 0.0247102
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068353, 0.0068652
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062044, 0.0062316
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231538, 0.0232551
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0180994, 0.0180207
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015547, 0.0015615

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032933, upper bound: 0.0033477
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032873, upper bound: 0.0033622
time: 2.05 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031738, upper bound: 0.0031256
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031585, upper bound: 0.0031403
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032488, upper bound: 0.0032373
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032385, upper bound: 0.0032512
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032989, upper bound: 0.0032045
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032511, upper bound: 0.0032502
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032610, upper bound: 0.0032037
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032410, upper bound: 0.0032267
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032266, upper bound: 0.0032340
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032266, upper bound: 0.0032339
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032639, upper bound: 0.0032740
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032639, upper bound: 0.0032740
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033581, upper bound: 0.0032977
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033568, upper bound: 0.0033004
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031284, upper bound: 0.0031449
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031303, upper bound: 0.0031444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0029540, upper bound: 0.0029424
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0029540, upper bound: 0.0029424
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032631, upper bound: 0.0032519
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032586, upper bound: 0.0032550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033224, upper bound: 0.0032648
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033106, upper bound: 0.0032695
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032598, upper bound: 0.0032531
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032438, upper bound: 0.0032625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032966, upper bound: 0.0032475
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032953, upper bound: 0.0032499
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0030286, upper bound: 0.0030114
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0030286, upper bound: 0.0030114
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033600, upper bound: 0.0033138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033595, upper bound: 0.0033163
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033553, upper bound: 0.0033314
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033568, upper bound: 0.0033313
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033631, upper bound: 0.0032937
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033511, upper bound: 0.0033016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033201, upper bound: 0.0033369
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033201, upper bound: 0.0033370
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032773, upper bound: 0.0031924
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032706, upper bound: 0.0031964
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032645, upper bound: 0.0032365
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032646, upper bound: 0.0032363
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0028768, upper bound: 0.0028754
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0028768, upper bound: 0.0028754
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032497, upper bound: 0.0032415
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032384, upper bound: 0.0032506
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033236, upper bound: 0.0032676
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033155, upper bound: 0.0032737
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032197, upper bound: 0.0032186
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032050, upper bound: 0.0032436
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033696, upper bound: 0.0033539
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033383, upper bound: 0.0033783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033366, upper bound: 0.0033074
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032809, upper bound: 0.0033590
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033272, upper bound: 0.0033132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033197, upper bound: 0.0033239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031560, upper bound: 0.0032026
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031614, upper bound: 0.0031978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0030527, upper bound: 0.0030651
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0030527, upper bound: 0.0030651
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032798, upper bound: 0.0032763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032660, upper bound: 0.0032925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032149, upper bound: 0.0031913
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031910, upper bound: 0.0032136
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032231, upper bound: 0.0032262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032191, upper bound: 0.0032466
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032779, upper bound: 0.0032592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032616, upper bound: 0.0032731
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032946, upper bound: 0.0032762
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032673, upper bound: 0.0032977
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033396, upper bound: 0.0033131
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033336, upper bound: 0.0033196
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033129, upper bound: 0.0033278
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033051, upper bound: 0.0033298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031971, upper bound: 0.0032068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0031971, upper bound: 0.0032068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0028019, upper bound: 0.0028041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0028019, upper bound: 0.0028041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033123, upper bound: 0.0033492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0033126, upper bound: 0.0033529
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032935, upper bound: 0.0033520
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032923, upper bound: 0.0033550
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032600, upper bound: 0.0033074
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032495, upper bound: 0.0033312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032933, upper bound: 0.0033477
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.46
Output dim: 5, lower bound: -0.0032873, upper bound: 0.0033622

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156776, 0.0158489
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044201, 0.0044684
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0326125, 0.0329690
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043157, 0.0043629
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246390, 0.0243726
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068454, 0.0067714
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062136, 0.0061464
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231880, 0.0229373
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178521, 0.0180473
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015570, 0.0015402

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032893, upper bound: 0.0031872
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032884, upper bound: 0.0031937
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154082, 0.0154304
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043441, 0.0043504
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0320521, 0.0320983
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042416, 0.0042477
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0239883, 0.0239538
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066647, 0.0066551
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060495, 0.0060408
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0225757, 0.0225432
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175454, 0.0175707
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015159, 0.0015137

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031603, upper bound: 0.0030982
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031613, upper bound: 0.0030956
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154030, 0.0154294
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043427, 0.0043501
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0320413, 0.0320962
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042402, 0.0042474
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0239867, 0.0239457
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066642, 0.0066528
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060491, 0.0060387
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0225742, 0.0225355
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175394, 0.0175695
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015158, 0.0015132

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033497, upper bound: 0.0032763
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033277, upper bound: 0.0032931
time: 2.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157358, 0.0158353
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044365, 0.0044646
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327337, 0.0329407
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043318, 0.0043592
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246178, 0.0244631
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068396, 0.0067966
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062083, 0.0061692
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231681, 0.0230225
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179185, 0.0180318
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015557, 0.0015459

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032981, upper bound: 0.0032060
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032401, upper bound: 0.0032337
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0157550, 0.0158160
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044419, 0.0044591
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0327736, 0.0329005
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043371, 0.0043539
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0245878, 0.0244930
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068312, 0.0068049
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062007, 0.0061768
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0231399, 0.0230506
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0179403, 0.0180098
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015538, 0.0015478

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032278, upper bound: 0.0031776
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032200, upper bound: 0.0031864
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0149568, 0.0150537
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0042169, 0.0042442
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0311131, 0.0313147
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0041173, 0.0041440
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0234026, 0.0232520
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0065019, 0.0064601
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0059018, 0.0058638
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0220245, 0.0218827
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0170314, 0.0171417
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0014789, 0.0014694

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032714, upper bound: 0.0032159
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032540, upper bound: 0.0032223
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0149807, 0.0150362
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0042236, 0.0042393
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0311629, 0.0312785
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0041239, 0.0041392
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0233756, 0.0232892
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0064944, 0.0064704
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0058950, 0.0058732
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0219990, 0.0219178
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0170586, 0.0171219
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0014772, 0.0014717

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032715, upper bound: 0.0031716
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032017, upper bound: 0.0032247
time: 2.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154544, 0.0155081
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043572, 0.0043723
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321484, 0.0322599
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042543, 0.0042691
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241090, 0.0240257
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066982, 0.0066751
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060799, 0.0060589
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226893, 0.0226109
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175981, 0.0176591
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015235, 0.0015183

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032303
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032342
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154455, 0.0155206
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043547, 0.0043758
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321297, 0.0322860
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042519, 0.0042725
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241286, 0.0240118
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067036, 0.0066712
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060849, 0.0060554
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227077, 0.0225977
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175879, 0.0176734
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015248, 0.0015174

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032806, upper bound: 0.0032214
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032735, upper bound: 0.0032381
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0153615, 0.0153951
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043310, 0.0043405
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0319551, 0.0320250
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042287, 0.0042380
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0239335, 0.0238812
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066494, 0.0066349
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060357, 0.0060225
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0225241, 0.0224749
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0174923, 0.0175305
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015124, 0.0015091

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032222, upper bound: 0.0032018
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032222, upper bound: 0.0032018
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0153668, 0.0153946
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043325, 0.0043403
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0319661, 0.0320239
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042302, 0.0042379
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0239327, 0.0238895
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066492, 0.0066372
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060355, 0.0060246
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0225233, 0.0224827
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0174983, 0.0175299
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015124, 0.0015097

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033363, upper bound: 0.0033099
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033355, upper bound: 0.0033108
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154968, 0.0155541
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043691, 0.0043853
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322364, 0.0323557
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042660, 0.0042818
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241806, 0.0240915
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067181, 0.0066933
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060980, 0.0060755
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227567, 0.0226728
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176463, 0.0177115
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015281, 0.0015224

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0032866, upper bound: 0.0031927
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032653, upper bound: 0.0032194
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0155014, 0.0155490
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043704, 0.0043839
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0322462, 0.0323452
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042673, 0.0042804
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241728, 0.0240988
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0067159, 0.0066954
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060960, 0.0060774
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0227493, 0.0226796
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0176516, 0.0177058
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015276, 0.0015229

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033268, upper bound: 0.0032726
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0033259, upper bound: 0.0032746
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154503, 0.0155028
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043560, 0.0043708
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321398, 0.0322490
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042532, 0.0042676
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0241009, 0.0240193
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066959, 0.0066733
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060779, 0.0060573
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226816, 0.0226048
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175934, 0.0176532
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015230, 0.0015179

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031177, upper bound: 0.0031393
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0031200, upper bound: 0.0031386
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0154532, 0.0155019
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0043568, 0.0043706
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0321459, 0.0322471
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0042540, 0.0042674
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0240995, 0.0240238
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0066956, 0.0066745
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0060775, 0.0060585
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0226803, 0.0226091
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0175967, 0.0176521
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015229, 0.0015182

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032161, upper bound: 0.0032317
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032161, upper bound: 0.0032319
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0175927, -0.0015266, -0.0175927, -0.0015266, -0.0156607, 0.0158577
1: -0.0078987, -0.0033691, -0.0078987, -0.0033691, -0.0044153, 0.0044709
2: -0.0197185, 0.0137022, -0.0197185, 0.0137022, -0.0325774, 0.0329873
3: -0.0009821, 0.0034406, -0.0009821, 0.0034406, -0.0043111, 0.0043653
4: -0.0041483, 0.0208282, -0.0041483, 0.0208282, -0.0246527, 0.0243463
5: 0.9943537, 1.0012929, 0.9943537, 1.0012929, -0.0068492, 0.0067641
6: 0.0027585, 0.0090572, 0.0027585, 0.0090572, -0.0062170, 0.0061398
7: -0.0130871, 0.0104186, -0.0130871, 0.0104186, -0.0232009, 0.0229126
8: -0.0173017, 0.0009929, -0.0173017, 0.0009929, -0.0178329, 0.0180573
9: -0.0040954, -0.0025170, -0.0040954, -0.0025170, -0.0015579, 0.0015385

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032208, upper bound: 0.0031239
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0032063, upper bound: 0.0031371
time: 2.09 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 5.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032893, upper bound: 0.0031872
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032884, upper bound: 0.0031937
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0031603, upper bound: 0.0030982
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0031613, upper bound: 0.0030956
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0033497, upper bound: 0.0032763
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0033277, upper bound: 0.0032931
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032981, upper bound: 0.0032060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032401, upper bound: 0.0032337
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032278, upper bound: 0.0031776
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032200, upper bound: 0.0031864
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032714, upper bound: 0.0032159
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032540, upper bound: 0.0032223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032715, upper bound: 0.0031716
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032017, upper bound: 0.0032247
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032725, upper bound: 0.0032303
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032629, upper bound: 0.0032342
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032806, upper bound: 0.0032214
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032735, upper bound: 0.0032381
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032222, upper bound: 0.0032018
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032222, upper bound: 0.0032018
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0033363, upper bound: 0.0033099
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0033355, upper bound: 0.0033108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032866, upper bound: 0.0031927
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032653, upper bound: 0.0032194
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0033268, upper bound: 0.0032726
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0033259, upper bound: 0.0032746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0031177, upper bound: 0.0031393
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0031200, upper bound: 0.0031386
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032161, upper bound: 0.0032317
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032161, upper bound: 0.0032319
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032208, upper bound: 0.0031239
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.41
Output dim: 5, lower bound: -0.0032063, upper bound: 0.0031371
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033236, upper bound: 0.0032676
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033155, upper bound: 0.0032737
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033696, upper bound: 0.0033539
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033383, upper bound: 0.0033783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033366, upper bound: 0.0033074
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032809, upper bound: 0.0033590
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033272, upper bound: 0.0033132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033197, upper bound: 0.0033239
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032798, upper bound: 0.0032763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032660, upper bound: 0.0032925
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032779, upper bound: 0.0032592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032946, upper bound: 0.0032762
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032673, upper bound: 0.0032977
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033396, upper bound: 0.0033131
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033336, upper bound: 0.0033196
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033129, upper bound: 0.0033278
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033051, upper bound: 0.0033298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033123, upper bound: 0.0033492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0033126, upper bound: 0.0033529
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032935, upper bound: 0.0033520
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032923, upper bound: 0.0033550
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032600, upper bound: 0.0033074
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032495, upper bound: 0.0033312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032933, upper bound: 0.0033477
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.41
Output dim: 5, lower bound: -0.0032873, upper bound: 0.0033622
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.006939222104847431
rel_dist={5: [-0.0035797029976590844, 0.0035794619380657977]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 1802.59 seconds
