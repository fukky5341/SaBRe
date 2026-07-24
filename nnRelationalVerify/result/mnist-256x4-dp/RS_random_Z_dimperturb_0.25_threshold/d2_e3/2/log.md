## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00059895


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009768, 0.0009768)
1: (-0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002434, 0.0002434)
2: (0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012899, 0.0012899)
3: (-0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005871, 0.0005871)
4: (0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002497, 0.0002497)
5: (0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0016224, 0.0016224)
6: (-0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0004118, 0.0004118)
7: (-0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010654, 0.0010654)
8: (-0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005603, 0.0005603)
9: (-0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006497, 0.0006497)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.43 = 2.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006574, upper bound: 0.0006575

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006329, upper bound: 0.0006424
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006424, upper bound: 0.0006329
time: 0.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 0, lower bound: -0.0006329, upper bound: 0.0006424
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 0, lower bound: -0.0006424, upper bound: 0.0006329

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009513, 0.0009542
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002370, 0.0002378
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012601, 0.0012562
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005718, 0.0005735
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002439, 0.0002431
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015848, 0.0015800
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0004010, 0.0004022
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010375, 0.0010407
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005456, 0.0005473
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006346, 0.0006327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006225, upper bound: 0.0006294
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006170, upper bound: 0.0006306
time: 0.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009542, 0.0009513
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002378, 0.0002370
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012562, 0.0012601
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005735, 0.0005718
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002431, 0.0002439
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015800, 0.0015848
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0004022, 0.0004010
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010407, 0.0010375
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005473, 0.0005456
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006327, 0.0006346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006225, upper bound: 0.0006148
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006248, upper bound: 0.0006128
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0006225, upper bound: 0.0006294
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0006170, upper bound: 0.0006306
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0006225, upper bound: 0.0006148
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0006248, upper bound: 0.0006128

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009328, 0.0009355
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002324, 0.0002331
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012353, 0.0012317
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005606, 0.0005622
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002391, 0.0002384
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015537, 0.0015492
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003932, 0.0003943
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010173, 0.0010203
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005350, 0.0005365
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006222, 0.0006204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006039, upper bound: 0.0006057
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006017, upper bound: 0.0006104
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009325, 0.0009358
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002324, 0.0002332
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012357, 0.0012314
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005605, 0.0005624
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002392, 0.0002383
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015541, 0.0015488
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003931, 0.0003945
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010171, 0.0010206
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005349, 0.0005367
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006223, 0.0006202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005992, upper bound: 0.0005653
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0006146
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009437, 0.0009418
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002352, 0.0002347
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012436, 0.0012462
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005672, 0.0005660
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002407, 0.0002412
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015641, 0.0015674
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003978, 0.0003970
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010293, 0.0010271
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005413, 0.0005401
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006263, 0.0006276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006054, upper bound: 0.0005530
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0005978
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009542, 0.0009408
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002378, 0.0002344
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012423, 0.0012601
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005735, 0.0005655
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002404, 0.0002439
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015625, 0.0015848
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0004022, 0.0003966
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010407, 0.0010261
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005473, 0.0005396
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006257, 0.0006346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005777, upper bound: 0.0005658
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005773, upper bound: 0.0005678
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0006039, upper bound: 0.0006057
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0006017, upper bound: 0.0006104
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0005992, upper bound: 0.0005653
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0006146
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0006054, upper bound: 0.0005530
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0005978
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0005777, upper bound: 0.0005658
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0005773, upper bound: 0.0005678

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009338, 0.0009319
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002327, 0.0002322
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012306, 0.0012331
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005613, 0.0005601
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002382, 0.0002387
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015478, 0.0015509
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003936, 0.0003928
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010185, 0.0010164
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005356, 0.0005345
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006198, 0.0006211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005892, upper bound: 0.0005953
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005922, upper bound: 0.0005868
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0009290, 0.0009365
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002315, 0.0002334
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0012367, 0.0012267
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005584, 0.0005629
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002394, 0.0002374
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0015554, 0.0015429
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003916, 0.0003948
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0010132, 0.0010214
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0005328, 0.0005371
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0006228, 0.0006178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005760, upper bound: 0.0005747
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005596, upper bound: 0.0005809
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0008482, 0.0008225
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002114, 0.0002049
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0010861, 0.0011201
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005098, 0.0004943
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002102, 0.0002168
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0013660, 0.0014088
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003576, 0.0003467
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0009251, 0.0008970
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0004865, 0.0004717
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0005470, 0.0005641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005487
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005781, upper bound: 0.0005486
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0008193, 0.0008524
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002041, 0.0002124
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0011256, 0.0010818
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0004924, 0.0005123
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002178, 0.0002094
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0014157, 0.0013607
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003453, 0.0003593
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0008935, 0.0009296
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0004699, 0.0004889
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0005669, 0.0005449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005455, upper bound: 0.0006039
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005455, upper bound: 0.0005955
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0008615, 0.0008299
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0002147, 0.0002068
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0010958, 0.0011375
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0005178, 0.0004988
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002121, 0.0002202
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0013783, 0.0014307
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003631, 0.0003498
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0009395, 0.0009051
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0004941, 0.0004760
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0005519, 0.0005729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005607, upper bound: 0.0005062
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005601, upper bound: 0.0005087
time: 0.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005892, upper bound: 0.0005953
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005922, upper bound: 0.0005868
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005760, upper bound: 0.0005747
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005596, upper bound: 0.0005809
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005487
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005781, upper bound: 0.0005486
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005455, upper bound: 0.0006039
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005455, upper bound: 0.0005955
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005607, upper bound: 0.0005062
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0005601, upper bound: 0.0005087

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9906939, 0.9926089, 0.9906939, 0.9926089, -0.0007923, 0.0008285
1: -0.0035828, -0.0031056, -0.0035828, -0.0031056, -0.0001974, 0.0002064
2: 0.0064042, 0.0089329, 0.0064042, 0.0089329, -0.0010941, 0.0010463
3: -0.0053390, -0.0041880, -0.0053390, -0.0041880, -0.0004762, 0.0004980
4: 0.0017674, 0.0022568, 0.0017674, 0.0022568, -0.0002118, 0.0002025
5: 0.0070143, 0.0101947, 0.0070143, 0.0101947, -0.0013760, 0.0013160
6: -0.0010467, -0.0002395, -0.0010467, -0.0002395, -0.0003340, 0.0003493
7: -0.0058457, -0.0037572, -0.0058457, -0.0037572, -0.0008642, 0.0009036
8: -0.0026384, -0.0015400, -0.0026384, -0.0015400, -0.0004545, 0.0004752
9: -0.0000781, 0.0011955, -0.0000781, 0.0011955, -0.0005510, 0.0005270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005251, upper bound: 0.0005853
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005252, upper bound: 0.0005808
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.48
Output dim: 0, lower bound: -0.0005251, upper bound: 0.0005853
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.48
Output dim: 0, lower bound: -0.0005252, upper bound: 0.0005808

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.79 + 30.35 = 33.15 seconds
