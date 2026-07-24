## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06656274


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843)
1: (-0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329)
2: (0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880)
3: (-0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0514377, 0.0514377)
4: (-0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099)
5: (-0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069)
6: (-0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348)
7: (0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089)
8: (-0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388)
9: (-0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 1.49 = 2.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0747682, upper bound: 0.0747682

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0747233, upper bound: 0.0746864
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0746864, upper bound: 0.0747233
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 7, lower bound: -0.0747233, upper bound: 0.0746864
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 7, lower bound: -0.0746864, upper bound: 0.0747233

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0513658, 0.0513961
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0731082, upper bound: 0.0735374
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0735707, upper bound: 0.0730314
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0513961, 0.0513658
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0730314, upper bound: 0.0735707
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0735374, upper bound: 0.0731082
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 7, lower bound: -0.0731082, upper bound: 0.0735374
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 7, lower bound: -0.0735707, upper bound: 0.0730314
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 7, lower bound: -0.0730314, upper bound: 0.0735707
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 7, lower bound: -0.0735374, upper bound: 0.0731082

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0508339, 0.0507974
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724282, upper bound: 0.0733417
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0729232, upper bound: 0.0729770
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0507672, 0.0508692
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0729799, upper bound: 0.0728451
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0733726, upper bound: 0.0723388
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0508692, 0.0507672
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0723388, upper bound: 0.0733726
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0728451, upper bound: 0.0729799
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0507974, 0.0508339
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0729770, upper bound: 0.0729232
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0733417, upper bound: 0.0724282
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0724282, upper bound: 0.0733417
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0729232, upper bound: 0.0729770
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0729799, upper bound: 0.0728451
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0733726, upper bound: 0.0723388
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0723388, upper bound: 0.0733726
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0728451, upper bound: 0.0729799
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0729770, upper bound: 0.0729232
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 7, lower bound: -0.0733417, upper bound: 0.0724282

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0505269, 0.0503565
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0719484, upper bound: 0.0724620
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0712741, upper bound: 0.0728451
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503929, 0.0504813
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724317, upper bound: 0.0721268
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0718195, upper bound: 0.0724965
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504531, 0.0504283
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724985, upper bound: 0.0717824
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0721386, upper bound: 0.0723522
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503262, 0.0505593
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0728789, upper bound: 0.0712499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724673, upper bound: 0.0718564
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0505593, 0.0503262
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0718564, upper bound: 0.0724673
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0712499, upper bound: 0.0728789
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504283, 0.0504531
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0723522, upper bound: 0.0721386
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0717824, upper bound: 0.0724985
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504812, 0.0503929
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724965, upper bound: 0.0718195
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0721268, upper bound: 0.0724317
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503565, 0.0505269
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0728451, upper bound: 0.0712741
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724620, upper bound: 0.0719484
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0719484, upper bound: 0.0724620
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0712741, upper bound: 0.0728451
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0724317, upper bound: 0.0721268
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0718195, upper bound: 0.0724965
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0724985, upper bound: 0.0717824
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0721386, upper bound: 0.0723522
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0728789, upper bound: 0.0712499
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0724673, upper bound: 0.0718564
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0718564, upper bound: 0.0724673
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0712499, upper bound: 0.0728789
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0723522, upper bound: 0.0721386
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0717824, upper bound: 0.0724985
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0724965, upper bound: 0.0718195
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0721268, upper bound: 0.0724317
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0728451, upper bound: 0.0712741
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0724620, upper bound: 0.0719484

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504111, 0.0503198
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0718948, upper bound: 0.0675680
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678564, upper bound: 0.0724094
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504868, 0.0502407
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0712197, upper bound: 0.0675607
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678440, upper bound: 0.0727930
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502771, 0.0504571
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0723823, upper bound: 0.0679500
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0674282, upper bound: 0.0720729
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503456, 0.0503654
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0717689, upper bound: 0.0679508
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0674097, upper bound: 0.0724437
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503372, 0.0503795
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724458, upper bound: 0.0673564
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679495, upper bound: 0.0717318
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504275, 0.0503125
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0720846, upper bound: 0.0673571
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679493, upper bound: 0.0723027
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502104, 0.0505190
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0728271, upper bound: 0.0677637
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675630, upper bound: 0.0711954
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502890, 0.0504435
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724150, upper bound: 0.0677970
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675703, upper bound: 0.0718029
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504435, 0.0502890
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0718029, upper bound: 0.0675703
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677970, upper bound: 0.0724150
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0505190, 0.0502104
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0711954, upper bound: 0.0675630
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677637, upper bound: 0.0728270
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503125, 0.0504274
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0723027, upper bound: 0.0679493
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673571, upper bound: 0.0720846
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503795, 0.0503372
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0717318, upper bound: 0.0679495
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673564, upper bound: 0.0724458
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503654, 0.0503455
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724437, upper bound: 0.0674097
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679508, upper bound: 0.0717689
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504571, 0.0502771
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0720729, upper bound: 0.0674282
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679500, upper bound: 0.0723823
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502407, 0.0504868
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.38 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0727930, upper bound: 0.0678440
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675607, upper bound: 0.0712197
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503198, 0.0504111
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0724094, upper bound: 0.0678564
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675680, upper bound: 0.0718948
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0718948, upper bound: 0.0675680
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0678564, upper bound: 0.0724094
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0712197, upper bound: 0.0675607
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0678440, upper bound: 0.0727930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0723823, upper bound: 0.0679500
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0674282, upper bound: 0.0720729
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0717689, upper bound: 0.0679508
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0674097, upper bound: 0.0724437
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0724458, upper bound: 0.0673564
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0679495, upper bound: 0.0717318
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0720846, upper bound: 0.0673571
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0679493, upper bound: 0.0723027
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0728271, upper bound: 0.0677637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0675630, upper bound: 0.0711954
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0724150, upper bound: 0.0677970
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0675703, upper bound: 0.0718029
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0718029, upper bound: 0.0675703
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0677970, upper bound: 0.0724150
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0711954, upper bound: 0.0675630
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0677637, upper bound: 0.0728270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0723027, upper bound: 0.0679493
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0673571, upper bound: 0.0720846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0717318, upper bound: 0.0679495
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0673564, upper bound: 0.0724458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0724437, upper bound: 0.0674097
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0679508, upper bound: 0.0717689
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0720729, upper bound: 0.0674282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0679500, upper bound: 0.0723823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0727930, upper bound: 0.0678440
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0675607, upper bound: 0.0712197
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0724094, upper bound: 0.0678564
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 7, lower bound: -0.0675680, upper bound: 0.0718948

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502426, 0.0502426
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682115, upper bound: 0.0650921
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0690148, upper bound: 0.0647856
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503482, 0.0501514
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647660, upper bound: 0.0696050
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650753, upper bound: 0.0694647
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503184, 0.0501548
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678147, upper bound: 0.0651089
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0684353, upper bound: 0.0648005
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504281, 0.0500722
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647034, upper bound: 0.0699701
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650084, upper bound: 0.0698074
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501087, 0.0504034
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0698884, upper bound: 0.0651323
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0699822, upper bound: 0.0647670
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501906, 0.0502887
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647833, upper bound: 0.0687394
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650431, upper bound: 0.0679104
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501771, 0.0503037
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694006, upper bound: 0.0651490
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694650, upper bound: 0.0647973
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502710, 0.0501970
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647132, upper bound: 0.0690584
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650015, upper bound: 0.0682092
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501688, 0.0503018
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682231, upper bound: 0.0649829
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0691663, upper bound: 0.0647167
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502765, 0.0502111
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647975, upper bound: 0.0694485
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651464, upper bound: 0.0693943
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502590, 0.0502235
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679064, upper bound: 0.0650251
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687537, upper bound: 0.0647825
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503746, 0.0501440
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647597, upper bound: 0.0698178
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651277, upper bound: 0.0697771
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500420, 0.0504596
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0699238, upper bound: 0.0649989
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0700663, upper bound: 0.0647061
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501252, 0.0503505
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0648005, upper bound: 0.0684489
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651077, upper bound: 0.0678185
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501205, 0.0503785
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694665, upper bound: 0.0650576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0696248, upper bound: 0.0647653
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502134, 0.0502751
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647840, upper bound: 0.0689875
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650908, upper bound: 0.0681969
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502751, 0.0502134
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681969, upper bound: 0.0650908
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0689875, upper bound: 0.0647841
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503785, 0.0501205
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647653, upper bound: 0.0696248
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650576, upper bound: 0.0694665
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503505, 0.0501252
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0678185, upper bound: 0.0651077
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0684489, upper bound: 0.0648005
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504597, 0.0500420
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647061, upper bound: 0.0700663
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0649989, upper bound: 0.0699238
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501440, 0.0503746
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0697771, upper bound: 0.0651277
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0698178, upper bound: 0.0647597
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502235, 0.0502590
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647824, upper bound: 0.0687537
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650251, upper bound: 0.0679064
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502111, 0.0502765
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0693943, upper bound: 0.0651464
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694485, upper bound: 0.0647975
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503018, 0.0501688
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647167, upper bound: 0.0691663
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0649829, upper bound: 0.0682231
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501970, 0.0502710
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682092, upper bound: 0.0650015
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0690584, upper bound: 0.0647132
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503037, 0.0501771
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647973, upper bound: 0.0694650
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651490, upper bound: 0.0694006
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502887, 0.0501906
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679104, upper bound: 0.0650431
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687394, upper bound: 0.0647834
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0504034, 0.0501087
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647670, upper bound: 0.0699822
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651323, upper bound: 0.0698884
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500722, 0.0504281
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0698074, upper bound: 0.0650084
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0699701, upper bound: 0.0647034
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501548, 0.0503184
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0648005, upper bound: 0.0684353
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651089, upper bound: 0.0678147
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501514, 0.0503482
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0694647, upper bound: 0.0650753
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0696050, upper bound: 0.0647660
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502426, 0.0502426
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647856, upper bound: 0.0690148
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650921, upper bound: 0.0682115
time: 0.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0682115, upper bound: 0.0650921
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0690148, upper bound: 0.0647856
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647660, upper bound: 0.0696050
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650753, upper bound: 0.0694647
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0678147, upper bound: 0.0651089
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0684353, upper bound: 0.0648005
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647034, upper bound: 0.0699701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650084, upper bound: 0.0698074
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0698884, upper bound: 0.0651323
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0699822, upper bound: 0.0647670
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647833, upper bound: 0.0687394
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650431, upper bound: 0.0679104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0694006, upper bound: 0.0651490
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0694650, upper bound: 0.0647973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647132, upper bound: 0.0690584
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650015, upper bound: 0.0682092
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0682231, upper bound: 0.0649829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0691663, upper bound: 0.0647167
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647975, upper bound: 0.0694485
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0651464, upper bound: 0.0693943
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0679064, upper bound: 0.0650251
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0687537, upper bound: 0.0647825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647597, upper bound: 0.0698178
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0651277, upper bound: 0.0697771
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0699238, upper bound: 0.0649989
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0700663, upper bound: 0.0647061
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0648005, upper bound: 0.0684489
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0651077, upper bound: 0.0678185
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0694665, upper bound: 0.0650576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0696248, upper bound: 0.0647653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647840, upper bound: 0.0689875
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650908, upper bound: 0.0681969
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0681969, upper bound: 0.0650908
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0689875, upper bound: 0.0647841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647653, upper bound: 0.0696248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650576, upper bound: 0.0694665
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0678185, upper bound: 0.0651077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0684489, upper bound: 0.0648005
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647061, upper bound: 0.0700663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0649989, upper bound: 0.0699238
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0697771, upper bound: 0.0651277
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0698178, upper bound: 0.0647597
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647824, upper bound: 0.0687537
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650251, upper bound: 0.0679064
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0693943, upper bound: 0.0651464
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0694485, upper bound: 0.0647975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647167, upper bound: 0.0691663
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0649829, upper bound: 0.0682231
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0682092, upper bound: 0.0650015
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0690584, upper bound: 0.0647132
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647973, upper bound: 0.0694650
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0651490, upper bound: 0.0694006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0679104, upper bound: 0.0650431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0687394, upper bound: 0.0647834
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647670, upper bound: 0.0699822
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0651323, upper bound: 0.0698884
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0698074, upper bound: 0.0650084
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0699701, upper bound: 0.0647034
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0648005, upper bound: 0.0684353
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0651089, upper bound: 0.0678147
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0694647, upper bound: 0.0650753
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0696050, upper bound: 0.0647660
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0647856, upper bound: 0.0690148
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 7, lower bound: -0.0650921, upper bound: 0.0682115

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502707, 0.0501121
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0609908, upper bound: 0.0641618
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672785, upper bound: 0.0605333
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502033, 0.0503198
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0609008, upper bound: 0.0637483
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682702, upper bound: 0.0605333
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502707, 0.0501121
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604302, upper bound: 0.0687912
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636928, upper bound: 0.0612747
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502033, 0.0503198
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604244, upper bound: 0.0686059
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0641295, upper bound: 0.0614412
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503457, 0.0500330
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604937, upper bound: 0.0641725
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669324, upper bound: 0.0606839
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502791, 0.0502407
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604530, upper bound: 0.0637584
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676609, upper bound: 0.0606839
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503457, 0.0500330
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0602278, upper bound: 0.0691541
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636214, upper bound: 0.0615380
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502791, 0.0502407
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0602277, upper bound: 0.0689345
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640670, upper bound: 0.0616213
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501841, 0.0502494
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613756, upper bound: 0.0642526
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0689963, upper bound: 0.0604834
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500694, 0.0504571
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0610871, upper bound: 0.0637349
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0691448, upper bound: 0.0604834
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501841, 0.0502494
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604310, upper bound: 0.0679641
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636981, upper bound: 0.0608553
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500694, 0.0504571
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604244, upper bound: 0.0670333
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640573, upper bound: 0.0608553
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502549, 0.0501577
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0609886, upper bound: 0.0642537
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0685179, upper bound: 0.0606762
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501378, 0.0503654
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0607426, upper bound: 0.0637519
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686583, upper bound: 0.0606762
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502549, 0.0501577
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0602693, upper bound: 0.0682853
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636214, upper bound: 0.0612325
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501378, 0.0503654
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0602448, upper bound: 0.0672816
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640274, upper bound: 0.0612330
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501711, 0.0501718
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613398, upper bound: 0.0640187
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672910, upper bound: 0.0601769
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501295, 0.0503795
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613375, upper bound: 0.0636352
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683988, upper bound: 0.0601906
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501711, 0.0501718
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0607521, upper bound: 0.0686266
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637451, upper bound: 0.0606547
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501295, 0.0503795
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0607521, upper bound: 0.0685179
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642400, upper bound: 0.0608764
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502486, 0.0501048
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0609323, upper bound: 0.0640395
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670220, upper bound: 0.0603231
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502197, 0.0503125
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0609323, upper bound: 0.0636983
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679779, upper bound: 0.0603485
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502486, 0.0501048
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0605065, upper bound: 0.0689863
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637160, upper bound: 0.0609667
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502197, 0.0503125
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0605065, upper bound: 0.0688872
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642232, upper bound: 0.0611882
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500947, 0.0503113
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0617539, upper bound: 0.0640639
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0690275, upper bound: 0.0601519
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500027, 0.0505190
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0616028, upper bound: 0.0636347
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0692632, upper bound: 0.0601537
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500947, 0.0503113
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0607673, upper bound: 0.0676765
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637504, upper bound: 0.0603597
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500027, 0.0505190
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0607673, upper bound: 0.0669512
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0641642, upper bound: 0.0604029
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501751, 0.0502358
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0614910, upper bound: 0.0641275
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686059, upper bound: 0.0603196
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500813, 0.0504435
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613388, upper bound: 0.0636929
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687969, upper bound: 0.0603343
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501751, 0.0502358
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0605482, upper bound: 0.0682232
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637229, upper bound: 0.0607220
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500813, 0.0504435
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0605482, upper bound: 0.0672656
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0641548, upper bound: 0.0607965
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503007, 0.0500812
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0607965, upper bound: 0.0641548
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672656, upper bound: 0.0605482
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502358, 0.0502890
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0607220, upper bound: 0.0637229
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682232, upper bound: 0.0605482
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503007, 0.0500812
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0603343, upper bound: 0.0687969
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636929, upper bound: 0.0613388
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502358, 0.0502890
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0603196, upper bound: 0.0686059
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0641275, upper bound: 0.0614910
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503747, 0.0500027
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604029, upper bound: 0.0641642
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669512, upper bound: 0.0607673
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503113, 0.0502104
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0603597, upper bound: 0.0637504
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676765, upper bound: 0.0607673
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503747, 0.0500027
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0601537, upper bound: 0.0692632
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636347, upper bound: 0.0616028
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503113, 0.0502104
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0601519, upper bound: 0.0690275
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640639, upper bound: 0.0617539
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502119, 0.0502197
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0611882, upper bound: 0.0642232
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0688872, upper bound: 0.0605065
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501048, 0.0504274
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0609667, upper bound: 0.0637160
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0689863, upper bound: 0.0605065
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502119, 0.0502197
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0603485, upper bound: 0.0679779
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636983, upper bound: 0.0609323
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501048, 0.0504274
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0603231, upper bound: 0.0670220
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640395, upper bound: 0.0609323
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502835, 0.0501295
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0608764, upper bound: 0.0642400
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0685179, upper bound: 0.0607521
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501718, 0.0503372
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0606547, upper bound: 0.0637451
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686266, upper bound: 0.0607521
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502835, 0.0501295
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0601906, upper bound: 0.0683988
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0636352, upper bound: 0.0613375
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501718, 0.0503372
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0601769, upper bound: 0.0672910
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0640187, upper bound: 0.0613398
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501997, 0.0501378
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612330, upper bound: 0.0640274
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672816, upper bound: 0.0602448
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501577, 0.0503455
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612325, upper bound: 0.0636214
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682853, upper bound: 0.0602693
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501997, 0.0501378
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606762, upper bound: 0.0686583
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637519, upper bound: 0.0607426
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501577, 0.0503455
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606762, upper bound: 0.0685179
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642537, upper bound: 0.0609886
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502764, 0.0500694
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0608553, upper bound: 0.0640573
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670333, upper bound: 0.0604244
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502494, 0.0502771
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0608553, upper bound: 0.0636981
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679641, upper bound: 0.0604310
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502764, 0.0500694
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604834, upper bound: 0.0691448
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637349, upper bound: 0.0610871
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502494, 0.0502771
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604834, upper bound: 0.0689963
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0642526, upper bound: 0.0613756
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501225, 0.0502791
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0616213, upper bound: 0.0640670
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0689345, upper bound: 0.0602277
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500330, 0.0504868
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615380, upper bound: 0.0636214
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0691541, upper bound: 0.0602278
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501225, 0.0502791
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606839, upper bound: 0.0676609
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637584, upper bound: 0.0604530
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500330, 0.0504868
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0606839, upper bound: 0.0669324
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0641725, upper bound: 0.0604937
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502037, 0.0502033
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0614412, upper bound: 0.0641295
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686059, upper bound: 0.0604244
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501121, 0.0504111
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612747, upper bound: 0.0636928
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687912, upper bound: 0.0604302
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502037, 0.0502033
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0605333, upper bound: 0.0682702
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0637483, upper bound: 0.0609008
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501121, 0.0504111
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0605333, upper bound: 0.0672785
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0641618, upper bound: 0.0609908
time: 0.61 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0609908, upper bound: 0.0641618
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0672785, upper bound: 0.0605333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0609008, upper bound: 0.0637483
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0682702, upper bound: 0.0605333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604302, upper bound: 0.0687912
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636928, upper bound: 0.0612747
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604244, upper bound: 0.0686059
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0641295, upper bound: 0.0614412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604937, upper bound: 0.0641725
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0669324, upper bound: 0.0606839
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604530, upper bound: 0.0637584
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0676609, upper bound: 0.0606839
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0602278, upper bound: 0.0691541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636214, upper bound: 0.0615380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0602277, upper bound: 0.0689345
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0640670, upper bound: 0.0616213
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0613756, upper bound: 0.0642526
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0689963, upper bound: 0.0604834
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0610871, upper bound: 0.0637349
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0691448, upper bound: 0.0604834
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604310, upper bound: 0.0679641
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636981, upper bound: 0.0608553
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604244, upper bound: 0.0670333
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0640573, upper bound: 0.0608553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0609886, upper bound: 0.0642537
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0685179, upper bound: 0.0606762
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0607426, upper bound: 0.0637519
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0686583, upper bound: 0.0606762
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0602693, upper bound: 0.0682853
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636214, upper bound: 0.0612325
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0602448, upper bound: 0.0672816
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0640274, upper bound: 0.0612330
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0613398, upper bound: 0.0640187
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0672910, upper bound: 0.0601769
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0613375, upper bound: 0.0636352
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0683988, upper bound: 0.0601906
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0607521, upper bound: 0.0686266
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637451, upper bound: 0.0606547
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0607521, upper bound: 0.0685179
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0642400, upper bound: 0.0608764
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0609323, upper bound: 0.0640395
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0670220, upper bound: 0.0603231
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0609323, upper bound: 0.0636983
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0679779, upper bound: 0.0603485
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0605065, upper bound: 0.0689863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637160, upper bound: 0.0609667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0605065, upper bound: 0.0688872
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0642232, upper bound: 0.0611882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0617539, upper bound: 0.0640639
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0690275, upper bound: 0.0601519
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0616028, upper bound: 0.0636347
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0692632, upper bound: 0.0601537
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0607673, upper bound: 0.0676765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637504, upper bound: 0.0603597
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0607673, upper bound: 0.0669512
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0641642, upper bound: 0.0604029
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0614910, upper bound: 0.0641275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0686059, upper bound: 0.0603196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0613388, upper bound: 0.0636929
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0687969, upper bound: 0.0603343
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0605482, upper bound: 0.0682232
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637229, upper bound: 0.0607220
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0605482, upper bound: 0.0672656
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0641548, upper bound: 0.0607965
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0607965, upper bound: 0.0641548
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0672656, upper bound: 0.0605482
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0607220, upper bound: 0.0637229
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0682232, upper bound: 0.0605482
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0603343, upper bound: 0.0687969
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636929, upper bound: 0.0613388
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0603196, upper bound: 0.0686059
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0641275, upper bound: 0.0614910
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604029, upper bound: 0.0641642
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0669512, upper bound: 0.0607673
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0603597, upper bound: 0.0637504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0676765, upper bound: 0.0607673
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0601537, upper bound: 0.0692632
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636347, upper bound: 0.0616028
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0601519, upper bound: 0.0690275
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0640639, upper bound: 0.0617539
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0611882, upper bound: 0.0642232
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0688872, upper bound: 0.0605065
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0609667, upper bound: 0.0637160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0689863, upper bound: 0.0605065
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0603485, upper bound: 0.0679779
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636983, upper bound: 0.0609323
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0603231, upper bound: 0.0670220
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0640395, upper bound: 0.0609323
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0608764, upper bound: 0.0642400
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0685179, upper bound: 0.0607521
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0606547, upper bound: 0.0637451
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0686266, upper bound: 0.0607521
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0601906, upper bound: 0.0683988
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0636352, upper bound: 0.0613375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0601769, upper bound: 0.0672910
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0640187, upper bound: 0.0613398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0612330, upper bound: 0.0640274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0672816, upper bound: 0.0602448
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0612325, upper bound: 0.0636214
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0682853, upper bound: 0.0602693
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0606762, upper bound: 0.0686583
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637519, upper bound: 0.0607426
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0606762, upper bound: 0.0685179
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0642537, upper bound: 0.0609886
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0608553, upper bound: 0.0640573
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0670333, upper bound: 0.0604244
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0608553, upper bound: 0.0636981
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0679641, upper bound: 0.0604310
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604834, upper bound: 0.0691448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637349, upper bound: 0.0610871
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0604834, upper bound: 0.0689963
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0642526, upper bound: 0.0613756
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0616213, upper bound: 0.0640670
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0689345, upper bound: 0.0602277
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0615380, upper bound: 0.0636214
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0691541, upper bound: 0.0602278
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0606839, upper bound: 0.0676609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637584, upper bound: 0.0604530
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0606839, upper bound: 0.0669324
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0641725, upper bound: 0.0604937
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0614412, upper bound: 0.0641295
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0686059, upper bound: 0.0604244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0612747, upper bound: 0.0636928
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0687912, upper bound: 0.0604302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0605333, upper bound: 0.0682702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0637483, upper bound: 0.0609008
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0605333, upper bound: 0.0672785
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 7, lower bound: -0.0641618, upper bound: 0.0609908

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458989, 0.0473747
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0646402, upper bound: 0.0603228
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670347, upper bound: 0.0574224
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458989, 0.0473747
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0653442, upper bound: 0.0603228
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680269, upper bound: 0.0574831
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473884, 0.0458077
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574430, upper bound: 0.0685355
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0602199, upper bound: 0.0665453
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473884, 0.0458077
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574174, upper bound: 0.0683520
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0602177, upper bound: 0.0664839
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459747, 0.0472812
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0645434, upper bound: 0.0604783
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666831, upper bound: 0.0574403
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459747, 0.0472812
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0652846, upper bound: 0.0604783
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0674093, upper bound: 0.0575122
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474608, 0.0457285
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574123, upper bound: 0.0689012
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0600029, upper bound: 0.0666514
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474608, 0.0457285
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0573829, upper bound: 0.0686819
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0600026, upper bound: 0.0666076
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0457650, 0.0473969
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667704, upper bound: 0.0602766
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687413, upper bound: 0.0572175
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0457650, 0.0473969
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667945, upper bound: 0.0602766
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0688872, upper bound: 0.0572199
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473661, 0.0459450
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575938, upper bound: 0.0677160
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0602201, upper bound: 0.0652524
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473661, 0.0459450
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574984, upper bound: 0.0667837
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0602171, upper bound: 0.0645438
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458334, 0.0473096
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0665765, upper bound: 0.0604712
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682596, upper bound: 0.0572504
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458334, 0.0473096
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666070, upper bound: 0.0604712
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683947, upper bound: 0.0572504
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474404, 0.0458533
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575660, upper bound: 0.0680476
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0600336, upper bound: 0.0653160
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474404, 0.0458533
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574739, upper bound: 0.0670418
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0600184, upper bound: 0.0646173
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458251, 0.0474422
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0646326, upper bound: 0.0599511
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670496, upper bound: 0.0574314
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458251, 0.0474422
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0653448, upper bound: 0.0599612
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0681566, upper bound: 0.0575140
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473081, 0.0458674
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574136, upper bound: 0.0683667
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0605571, upper bound: 0.0665696
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473081, 0.0458674
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0573782, upper bound: 0.0682590
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0605571, upper bound: 0.0665425
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459153, 0.0473678
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0645459, upper bound: 0.0601216
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667726, upper bound: 0.0574555
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459153, 0.0473678
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0652846, upper bound: 0.0601329
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677276, upper bound: 0.0575519
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473950, 0.0458003
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0573787, upper bound: 0.0687350
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0603017, upper bound: 0.0667002
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473950, 0.0458003
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0573542, upper bound: 0.0686356
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0603017, upper bound: 0.0666679
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0456983, 0.0474627
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667140, upper bound: 0.0599253
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687740, upper bound: 0.0572667
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0456983, 0.0474627
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667456, upper bound: 0.0599253
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0690066, upper bound: 0.0572669
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0472796, 0.0460068
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575690, upper bound: 0.0674267
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0605680, upper bound: 0.0652484
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0472796, 0.0460068
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574843, upper bound: 0.0667018
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0605680, upper bound: 0.0645425
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0457768, 0.0473901
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0665134, upper bound: 0.0601154
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683522, upper bound: 0.0573142
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0457768, 0.0473901
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0665741, upper bound: 0.0601207
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0685431, upper bound: 0.0573142
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473728, 0.0459314
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575436, upper bound: 0.0679834
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0603361, upper bound: 0.0653143
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473728, 0.0459314
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574629, upper bound: 0.0670181
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0603361, upper bound: 0.0646345
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459314, 0.0473728
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0646345, upper bound: 0.0603361
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670181, upper bound: 0.0574629
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459314, 0.0473728
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0653143, upper bound: 0.0603361
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679834, upper bound: 0.0575436
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473901, 0.0457768
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0573142, upper bound: 0.0685431
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0601207, upper bound: 0.0665741
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473901, 0.0457768
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0573142, upper bound: 0.0683522
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0601154, upper bound: 0.0665134
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0460068, 0.0472796
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0645425, upper bound: 0.0605680
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667018, upper bound: 0.0574843
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0460068, 0.0472796
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0652484, upper bound: 0.0605680
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0674267, upper bound: 0.0575690
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474627, 0.0456983
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0572669, upper bound: 0.0690066
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0599253, upper bound: 0.0667456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474627, 0.0456983
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0572667, upper bound: 0.0687740
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0599253, upper bound: 0.0667140
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458003, 0.0473950
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666679, upper bound: 0.0603017
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686356, upper bound: 0.0573542
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458003, 0.0473950
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667002, upper bound: 0.0603017
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0687350, upper bound: 0.0573787
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473678, 0.0459153
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575519, upper bound: 0.0677276
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0601329, upper bound: 0.0652846
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473678, 0.0459153
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574555, upper bound: 0.0667726
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0601216, upper bound: 0.0645459
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458674, 0.0473081
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0665425, upper bound: 0.0605571
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682590, upper bound: 0.0573782
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458674, 0.0473081
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0665696, upper bound: 0.0605571
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683667, upper bound: 0.0574136
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474422, 0.0458251
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575140, upper bound: 0.0681566
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0599612, upper bound: 0.0653448
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0474422, 0.0458251
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574314, upper bound: 0.0670496
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0599511, upper bound: 0.0646326
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458533, 0.0474404
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0646173, upper bound: 0.0600184
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670418, upper bound: 0.0574739
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458533, 0.0474404
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0653160, upper bound: 0.0600336
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680476, upper bound: 0.0575660
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473096, 0.0458334
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0572504, upper bound: 0.0683947
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604712, upper bound: 0.0666070
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473096, 0.0458334
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0572504, upper bound: 0.0682596
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0604712, upper bound: 0.0665765
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459450, 0.0473661
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0645438, upper bound: 0.0602171
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667837, upper bound: 0.0574984
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0459450, 0.0473661
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0652524, upper bound: 0.0602201
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0677160, upper bound: 0.0575938
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473969, 0.0457650
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0572199, upper bound: 0.0688872
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0602766, upper bound: 0.0667945
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473969, 0.0457650
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0572175, upper bound: 0.0687413
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0602766, upper bound: 0.0667704
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0457285, 0.0474608
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666076, upper bound: 0.0600026
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0686819, upper bound: 0.0573829
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0457285, 0.0474608
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666514, upper bound: 0.0600029
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0689012, upper bound: 0.0574123
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0472813, 0.0459747
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0575122, upper bound: 0.0674093
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604783, upper bound: 0.0652846
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0472813, 0.0459747
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574403, upper bound: 0.0666831
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0604783, upper bound: 0.0645434
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458077, 0.0473884
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0664839, upper bound: 0.0602177
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683520, upper bound: 0.0574174
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0458077, 0.0473884
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0665453, upper bound: 0.0602199
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0685355, upper bound: 0.0574430
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473747, 0.0458989
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574831, upper bound: 0.0680269
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0603228, upper bound: 0.0653442
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0473747, 0.0458989
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0574224, upper bound: 0.0670347
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0603228, upper bound: 0.0646402
time: 0.59 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0646402, upper bound: 0.0603228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0670347, upper bound: 0.0574224
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0653442, upper bound: 0.0603228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0680269, upper bound: 0.0574831
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574430, upper bound: 0.0685355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0602199, upper bound: 0.0665453
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574174, upper bound: 0.0683520
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0602177, upper bound: 0.0664839
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0645434, upper bound: 0.0604783
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0666831, upper bound: 0.0574403
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0652846, upper bound: 0.0604783
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0674093, upper bound: 0.0575122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574123, upper bound: 0.0689012
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0600029, upper bound: 0.0666514
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0573829, upper bound: 0.0686819
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0600026, upper bound: 0.0666076
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667704, upper bound: 0.0602766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0687413, upper bound: 0.0572175
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667945, upper bound: 0.0602766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0688872, upper bound: 0.0572199
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0575938, upper bound: 0.0677160
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0602201, upper bound: 0.0652524
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574984, upper bound: 0.0667837
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0602171, upper bound: 0.0645438
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0665765, upper bound: 0.0604712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0682596, upper bound: 0.0572504
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0666070, upper bound: 0.0604712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0683947, upper bound: 0.0572504
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0575660, upper bound: 0.0680476
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0600336, upper bound: 0.0653160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574739, upper bound: 0.0670418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0600184, upper bound: 0.0646173
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0646326, upper bound: 0.0599511
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0670496, upper bound: 0.0574314
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0653448, upper bound: 0.0599612
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0681566, upper bound: 0.0575140
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574136, upper bound: 0.0683667
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0605571, upper bound: 0.0665696
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0573782, upper bound: 0.0682590
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0605571, upper bound: 0.0665425
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0645459, upper bound: 0.0601216
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667726, upper bound: 0.0574555
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0652846, upper bound: 0.0601329
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0677276, upper bound: 0.0575519
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0573787, upper bound: 0.0687350
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0603017, upper bound: 0.0667002
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0573542, upper bound: 0.0686356
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0603017, upper bound: 0.0666679
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667140, upper bound: 0.0599253
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0687740, upper bound: 0.0572667
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667456, upper bound: 0.0599253
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0690066, upper bound: 0.0572669
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0575690, upper bound: 0.0674267
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0605680, upper bound: 0.0652484
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574843, upper bound: 0.0667018
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0605680, upper bound: 0.0645425
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0665134, upper bound: 0.0601154
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0683522, upper bound: 0.0573142
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0665741, upper bound: 0.0601207
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0685431, upper bound: 0.0573142
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0575436, upper bound: 0.0679834
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0603361, upper bound: 0.0653143
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574629, upper bound: 0.0670181
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0603361, upper bound: 0.0646345
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0646345, upper bound: 0.0603361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0670181, upper bound: 0.0574629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0653143, upper bound: 0.0603361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0679834, upper bound: 0.0575436
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0573142, upper bound: 0.0685431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0601207, upper bound: 0.0665741
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0573142, upper bound: 0.0683522
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0601154, upper bound: 0.0665134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0645425, upper bound: 0.0605680
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667018, upper bound: 0.0574843
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0652484, upper bound: 0.0605680
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0674267, upper bound: 0.0575690
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0572669, upper bound: 0.0690066
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0599253, upper bound: 0.0667456
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0572667, upper bound: 0.0687740
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0599253, upper bound: 0.0667140
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0666679, upper bound: 0.0603017
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0686356, upper bound: 0.0573542
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667002, upper bound: 0.0603017
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0687350, upper bound: 0.0573787
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0575519, upper bound: 0.0677276
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0601329, upper bound: 0.0652846
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574555, upper bound: 0.0667726
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0601216, upper bound: 0.0645459
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0665425, upper bound: 0.0605571
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0682590, upper bound: 0.0573782
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0665696, upper bound: 0.0605571
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0683667, upper bound: 0.0574136
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0575140, upper bound: 0.0681566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0599612, upper bound: 0.0653448
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574314, upper bound: 0.0670496
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0599511, upper bound: 0.0646326
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0646173, upper bound: 0.0600184
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0670418, upper bound: 0.0574739
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0653160, upper bound: 0.0600336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0680476, upper bound: 0.0575660
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0572504, upper bound: 0.0683947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0604712, upper bound: 0.0666070
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0572504, upper bound: 0.0682596
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0604712, upper bound: 0.0665765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0645438, upper bound: 0.0602171
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0667837, upper bound: 0.0574984
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0652524, upper bound: 0.0602201
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0677160, upper bound: 0.0575938
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0572199, upper bound: 0.0688872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0602766, upper bound: 0.0667945
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0572175, upper bound: 0.0687413
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0602766, upper bound: 0.0667704
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0666076, upper bound: 0.0600026
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0686819, upper bound: 0.0573829
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0666514, upper bound: 0.0600029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0689012, upper bound: 0.0574123
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0575122, upper bound: 0.0674093
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0604783, upper bound: 0.0652846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574403, upper bound: 0.0666831
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0604783, upper bound: 0.0645434
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0664839, upper bound: 0.0602177
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0683520, upper bound: 0.0574174
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0665453, upper bound: 0.0602199
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0685355, upper bound: 0.0574430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574831, upper bound: 0.0680269
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0603228, upper bound: 0.0653442
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0574224, upper bound: 0.0670347
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 7, lower bound: -0.0603228, upper bound: 0.0646402

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500998, 0.0501287
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577768, upper bound: 0.0503966
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577768, upper bound: 0.0503933
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500998, 0.0501287
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577768, upper bound: 0.0503966
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577768, upper bound: 0.0503933
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502169, 0.0500086
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505209, upper bound: 0.0584580
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505209, upper bound: 0.0584580
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0502169, 0.0500086
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505209, upper bound: 0.0584580
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505209, upper bound: 0.0584580
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501757, 0.0500435
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577272, upper bound: 0.0503585
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577272, upper bound: 0.0503544
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501757, 0.0500435
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577272, upper bound: 0.0503585
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0577272, upper bound: 0.0503544
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503032, 0.0499295
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505298, upper bound: 0.0584861
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505298, upper bound: 0.0584861
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501757, 0.0500435
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0529034, upper bound: 0.0566791
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0529034, upper bound: 0.0566791
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0503032, 0.0499295
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505298, upper bound: 0.0584861
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0505298, upper bound: 0.0584861
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0501757, 0.0500435
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0529034, upper bound: 0.0566791
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0529034, upper bound: 0.0566791
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500918, 0.0501459
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0566924, upper bound: 0.0529964
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0566924, upper bound: 0.0529964
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0499659, 0.0502553
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0582679, upper bound: 0.0504263
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0582679, upper bound: 0.0504261
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500918, 0.0501459
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0566924, upper bound: 0.0529964
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0566924, upper bound: 0.0529964
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0499659, 0.0502553
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0582679, upper bound: 0.0504263
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0582679, upper bound: 0.0504261
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500918, 0.0501459
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 0
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 190
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 249

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 0

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0504575, upper bound: 0.0579357
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0504575, upper bound: 0.0579357
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0180193, 0.0134649, -0.0180193, 0.0134649, -0.0314843, 0.0314843
1: -0.0322378, 0.0117951, -0.0322378, 0.0117951, -0.0440329, 0.0440329
2: 0.0270358, 0.0669238, 0.0270358, 0.0669238, -0.0398880, 0.0398880
3: -0.0021559, 0.0559740, -0.0021559, 0.0559740, -0.0500918, 0.0501459
4: -0.0253884, 0.0252215, -0.0253884, 0.0252215, -0.0506099, 0.0506099
5: -0.0087964, 0.0410104, -0.0087964, 0.0410104, -0.0498069, 0.0498069
6: -0.0508792, -0.0052445, -0.0508792, -0.0052445, -0.0456348, 0.0456348
7: 0.8582754, 0.9712843, 0.8582754, 0.9712843, -0.1130089, 0.1130089
8: -0.0115519, 0.0452869, -0.0115519, 0.0452869, -0.0568388, 0.0568388
9: -0.0218013, 0.0262989, -0.0218013, 0.0262989, -0.0481002, 0.0481002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.97 + 598.00 = 600.97 seconds
