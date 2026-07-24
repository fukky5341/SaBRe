## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0010557


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008161, 0.0008161)
1: (0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019704, 0.0019704)
2: (-0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339)
3: (0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007882, 0.0007882)
4: (0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356)
5: (0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219)
6: (-0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358)
7: (-0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559)
8: (0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0040754, 0.0040754)
9: (-0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.24 + 2.36 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0012034, upper bound: 0.0012034

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011978, upper bound: 0.0011978
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011978, upper bound: 0.0011978
time: 0.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 1, lower bound: -0.0011978, upper bound: 0.0011978
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 1, lower bound: -0.0011978, upper bound: 0.0011978

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008160, 0.0008160
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019692, 0.0019690
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007872, 0.0007874
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0040468, 0.0040500
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011237, upper bound: 0.0011757
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011757, upper bound: 0.0011221
time: 1.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008160, 0.0008160
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019690, 0.0019692
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007874, 0.0007872
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0040500, 0.0040468
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011538, upper bound: 0.0011538
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011538, upper bound: 0.0011538
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 1, lower bound: -0.0011237, upper bound: 0.0011757
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 1, lower bound: -0.0011757, upper bound: 0.0011221
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 1, lower bound: -0.0011538, upper bound: 0.0011538
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 1, lower bound: -0.0011538, upper bound: 0.0011538

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008152, 0.0008149
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019547, 0.0019583
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007797, 0.0007772
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039150, 0.0038649
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011002, upper bound: 0.0011668
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011145, upper bound: 0.0011457
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008149, 0.0008152
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019583, 0.0019544
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007770, 0.0007797
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038616, 0.0039158
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011606, upper bound: 0.0011084
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011626, upper bound: 0.0011078
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008158, 0.0008158
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019663, 0.0019661
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007852, 0.0007854
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0040042, 0.0040068
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011380, upper bound: 0.0011495
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011495, upper bound: 0.0011380
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008158, 0.0008160
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019690, 0.0019665
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007855, 0.0007872
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0040101, 0.0040468
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011303, upper bound: 0.0011307
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011307, upper bound: 0.0011303
time: 0.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011002, upper bound: 0.0011668
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011145, upper bound: 0.0011457
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011606, upper bound: 0.0011084
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011626, upper bound: 0.0011078
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011380, upper bound: 0.0011495
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011495, upper bound: 0.0011380
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011303, upper bound: 0.0011307
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 1, lower bound: -0.0011307, upper bound: 0.0011303

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008148, 0.0008143
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019466, 0.0019528
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007760, 0.0007717
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039039, 0.0038189
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010858, upper bound: 0.0011536
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010866, upper bound: 0.0011517
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008146, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019489, 0.0019502
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007742, 0.0007733
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038690, 0.0038509
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010931, upper bound: 0.0011222
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010923, upper bound: 0.0011230
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019559, 0.0019520
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007753, 0.0007780
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038248, 0.0038790
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011317, upper bound: 0.0010895
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011413, upper bound: 0.0010872
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019560, 0.0019520
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007753, 0.0007781
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038249, 0.0038811
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011352, upper bound: 0.0010889
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011434, upper bound: 0.0010863
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008154, 0.0008152
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019582, 0.0019604
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007813, 0.0007798
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039892, 0.0039590
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011194, upper bound: 0.0011305
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011193, upper bound: 0.0011311
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008152, 0.0008154
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019605, 0.0019580
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007796, 0.0007813
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039564, 0.0039902
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011242, upper bound: 0.0011193
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011305, upper bound: 0.0011159
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008153, 0.0008155
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019618, 0.0019596
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007801, 0.0007818
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039477, 0.0039831
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011119, upper bound: 0.0011116
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011114, upper bound: 0.0011122
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008153, 0.0008155
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019620, 0.0019595
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007801, 0.0007819
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039469, 0.0039855
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011122, upper bound: 0.0011114
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011116, upper bound: 0.0011119
time: 0.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0010858, upper bound: 0.0011536
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0010866, upper bound: 0.0011517
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0010931, upper bound: 0.0011222
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0010923, upper bound: 0.0011230
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011317, upper bound: 0.0010895
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011413, upper bound: 0.0010872
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011352, upper bound: 0.0010889
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011434, upper bound: 0.0010863
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011194, upper bound: 0.0011305
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011193, upper bound: 0.0011311
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011242, upper bound: 0.0011193
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011305, upper bound: 0.0011159
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011119, upper bound: 0.0011116
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011114, upper bound: 0.0011122
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011122, upper bound: 0.0011114
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 1, lower bound: -0.0011116, upper bound: 0.0011119

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008146, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019441, 0.0019504
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007743, 0.0007699
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038676, 0.0037815
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010658, upper bound: 0.0011337
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010651, upper bound: 0.0011344
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008146, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019441, 0.0019503
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007742, 0.0007699
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038666, 0.0037813
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010664, upper bound: 0.0011315
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010660, upper bound: 0.0011322
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019419, 0.0019433
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007687, 0.0007677
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037794, 0.0037597
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010787, upper bound: 0.0011100
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010796, upper bound: 0.0011062
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019420, 0.0019430
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007684, 0.0007678
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037742, 0.0037613
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010721, upper bound: 0.0011015
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010709, upper bound: 0.0011015
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008148
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019529, 0.0019516
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007747, 0.0007756
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037488, 0.0037676
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011038, upper bound: 0.0010801
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011220, upper bound: 0.0010668
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008145, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019555, 0.0019490
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007729, 0.0007774
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037135, 0.0038030
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011094, upper bound: 0.0010776
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011320, upper bound: 0.0010662
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008148
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019531, 0.0019516
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007747, 0.0007757
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037489, 0.0037701
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011156, upper bound: 0.0010695
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011156, upper bound: 0.0010702
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008145, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019556, 0.0019489
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007728, 0.0007775
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037125, 0.0038050
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011213, upper bound: 0.0010650
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011213, upper bound: 0.0010664
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008149, 0.0008147
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019523, 0.0019551
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007777, 0.0007757
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039140, 0.0038750
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011058, upper bound: 0.0011201
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011085, upper bound: 0.0011194
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008149, 0.0008148
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019529, 0.0019546
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007773, 0.0007762
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0039064, 0.0038838
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010568, upper bound: 0.0011166
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011025, upper bound: 0.0010708
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008152, 0.0008152
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019577, 0.0019579
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007789, 0.0007788
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038765, 0.0038745
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011120, upper bound: 0.0011082
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011128, upper bound: 0.0011056
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008150, 0.0008154
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019603, 0.0019554
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007771, 0.0007806
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038425, 0.0039103
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010696, upper bound: 0.0010988
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011167, upper bound: 0.0010576
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008149, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019556, 0.0019540
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007765, 0.0007777
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038539, 0.0038808
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011008, upper bound: 0.0011009
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011015, upper bound: 0.0011003
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008148, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019563, 0.0019534
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007761, 0.0007782
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038453, 0.0038906
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011002, upper bound: 0.0011016
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011008, upper bound: 0.0011011
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008149, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019558, 0.0019540
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007765, 0.0007778
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038531, 0.0038843
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010958, upper bound: 0.0011068
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011076, upper bound: 0.0010961
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008148, 0.0008151
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019565, 0.0019533
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007760, 0.0007783
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038434, 0.0038930
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010957, upper bound: 0.0011074
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011069, upper bound: 0.0010961
time: 0.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010658, upper bound: 0.0011337
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010651, upper bound: 0.0011344
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010664, upper bound: 0.0011315
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010660, upper bound: 0.0011322
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010787, upper bound: 0.0011100
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010796, upper bound: 0.0011062
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010721, upper bound: 0.0011015
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010709, upper bound: 0.0011015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011038, upper bound: 0.0010801
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011220, upper bound: 0.0010668
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011094, upper bound: 0.0010776
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011320, upper bound: 0.0010662
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011156, upper bound: 0.0010695
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011156, upper bound: 0.0010702
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011213, upper bound: 0.0010650
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011213, upper bound: 0.0010664
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011058, upper bound: 0.0011201
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011085, upper bound: 0.0011194
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010568, upper bound: 0.0011166
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011025, upper bound: 0.0010708
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011120, upper bound: 0.0011082
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011128, upper bound: 0.0011056
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010696, upper bound: 0.0010988
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011167, upper bound: 0.0010576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011008, upper bound: 0.0011009
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011015, upper bound: 0.0011003
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011002, upper bound: 0.0011016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011008, upper bound: 0.0011011
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010958, upper bound: 0.0011068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011076, upper bound: 0.0010961
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0010957, upper bound: 0.0011074
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 1, lower bound: -0.0011069, upper bound: 0.0010961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019382, 0.0019450
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007705, 0.0007657
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037873, 0.0036925
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010436, upper bound: 0.0011047
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010436, upper bound: 0.0011047
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019388, 0.0019443
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007700, 0.0007662
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037769, 0.0037012
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010460, upper bound: 0.0011146
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010467, upper bound: 0.0011044
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019382, 0.0019450
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007705, 0.0007657
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037862, 0.0036920
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010449, upper bound: 0.0011035
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010449, upper bound: 0.0011035
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019388, 0.0019442
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007699, 0.0007662
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037758, 0.0037010
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010471, upper bound: 0.0011126
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010475, upper bound: 0.0011011
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019395, 0.0019410
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007670, 0.0007660
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037453, 0.0037244
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010582, upper bound: 0.0010907
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010609, upper bound: 0.0010866
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019396, 0.0019409
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007670, 0.0007660
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037441, 0.0037256
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010588, upper bound: 0.0010871
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010616, upper bound: 0.0010817
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019360, 0.0019375
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007646, 0.0007636
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036684, 0.0036480
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010475, upper bound: 0.0010792
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010475, upper bound: 0.0010792
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019366, 0.0019367
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007641, 0.0007640
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036578, 0.0036555
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010494, upper bound: 0.0010823
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010528, upper bound: 0.0010783
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008142
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019452, 0.0019461
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007706, 0.0007700
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037194, 0.0037062
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010825, upper bound: 0.0010589
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010818, upper bound: 0.0010598
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019476, 0.0019438
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007690, 0.0007717
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036874, 0.0037405
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011005, upper bound: 0.0010427
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011003, upper bound: 0.0010470
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019477, 0.0019435
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007688, 0.0007717
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036836, 0.0037416
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010894, upper bound: 0.0010577
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010894, upper bound: 0.0010586
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008146
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019503, 0.0019412
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007672, 0.0007735
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036521, 0.0037771
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011126, upper bound: 0.0010459
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011120, upper bound: 0.0010466
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008143
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019473, 0.0019464
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007710, 0.0007717
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036600, 0.0036724
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010850, upper bound: 0.0010449
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010850, upper bound: 0.0010449
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019479, 0.0019458
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007706, 0.0007721
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036512, 0.0036812
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010893, upper bound: 0.0010607
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011055, upper bound: 0.0010460
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019486, 0.0019422
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007678, 0.0007723
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036484, 0.0037374
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010867, upper bound: 0.0010391
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010867, upper bound: 0.0010391
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019489, 0.0019424
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007679, 0.0007724
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036509, 0.0037409
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010907, upper bound: 0.0010568
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011115, upper bound: 0.0010451
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019498, 0.0019527
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007760, 0.0007740
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038757, 0.0038364
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010819, upper bound: 0.0010957
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010817, upper bound: 0.0010956
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019498, 0.0019527
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007759, 0.0007740
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038754, 0.0038361
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010444, upper bound: 0.0011035
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010909, upper bound: 0.0010572
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019384, 0.0019436
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007695, 0.0007658
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037685, 0.0036962
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010424, upper bound: 0.0011051
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010438, upper bound: 0.0011040
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008137, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019422, 0.0019400
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007670, 0.0007685
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037188, 0.0037495
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010870, upper bound: 0.0010582
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010909, upper bound: 0.0010579
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008150, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019553, 0.0019554
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007771, 0.0007770
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038419, 0.0038406
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010931, upper bound: 0.0010897
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010930, upper bound: 0.0010897
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008150, 0.0008150
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019553, 0.0019554
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007771, 0.0007771
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038425, 0.0038412
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010893, upper bound: 0.0010818
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010892, upper bound: 0.0010821
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008142
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019458, 0.0019444
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007694, 0.0007704
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036948, 0.0037145
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010513, upper bound: 0.0010797
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010506, upper bound: 0.0010796
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019500, 0.0019409
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007670, 0.0007733
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036467, 0.0037716
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011041, upper bound: 0.0010448
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011048, upper bound: 0.0010431
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008148
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019531, 0.0019516
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007748, 0.0007759
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038174, 0.0038441
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010819, upper bound: 0.0010957
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010956, upper bound: 0.0010841
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008147, 0.0008148
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019532, 0.0019516
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007748, 0.0007760
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038176, 0.0038444
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010844, upper bound: 0.0010951
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010962, upper bound: 0.0010815
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008146, 0.0008149
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019539, 0.0019509
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007744, 0.0007764
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038084, 0.0038539
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010407, upper bound: 0.0010870
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010851, upper bound: 0.0010415
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008146, 0.0008149
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019538, 0.0019510
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007744, 0.0007764
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038089, 0.0038537
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010409, upper bound: 0.0010860
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010864, upper bound: 0.0010414
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008144, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019480, 0.0019484
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007724, 0.0007722
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038270, 0.0038255
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010332, upper bound: 0.0010922
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010789, upper bound: 0.0010480
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008146
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019504, 0.0019462
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007709, 0.0007739
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037957, 0.0038581
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010823, upper bound: 0.0010774
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010886, upper bound: 0.0010738
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008144, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019487, 0.0019480
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007721, 0.0007727
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0038207, 0.0038343
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010815, upper bound: 0.0010962
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010840, upper bound: 0.0010956
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008146
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019509, 0.0019455
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007704, 0.0007742
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037859, 0.0038655
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010951, upper bound: 0.0010845
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010957, upper bound: 0.0010819
time: 0.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010436, upper bound: 0.0011047
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010436, upper bound: 0.0011047
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010460, upper bound: 0.0011146
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010467, upper bound: 0.0011044
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010449, upper bound: 0.0011035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010449, upper bound: 0.0011035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010471, upper bound: 0.0011126
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010475, upper bound: 0.0011011
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010582, upper bound: 0.0010907
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010609, upper bound: 0.0010866
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010588, upper bound: 0.0010871
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010616, upper bound: 0.0010817
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010475, upper bound: 0.0010792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010475, upper bound: 0.0010792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010494, upper bound: 0.0010823
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010528, upper bound: 0.0010783
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010825, upper bound: 0.0010589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010818, upper bound: 0.0010598
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011005, upper bound: 0.0010427
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011003, upper bound: 0.0010470
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010894, upper bound: 0.0010577
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010894, upper bound: 0.0010586
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011126, upper bound: 0.0010459
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011120, upper bound: 0.0010466
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010850, upper bound: 0.0010449
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010850, upper bound: 0.0010449
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010893, upper bound: 0.0010607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011055, upper bound: 0.0010460
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010867, upper bound: 0.0010391
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010867, upper bound: 0.0010391
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010907, upper bound: 0.0010568
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011115, upper bound: 0.0010451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010819, upper bound: 0.0010957
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010817, upper bound: 0.0010956
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010444, upper bound: 0.0011035
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010909, upper bound: 0.0010572
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010424, upper bound: 0.0011051
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010438, upper bound: 0.0011040
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010870, upper bound: 0.0010582
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010909, upper bound: 0.0010579
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010931, upper bound: 0.0010897
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010930, upper bound: 0.0010897
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010893, upper bound: 0.0010818
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010892, upper bound: 0.0010821
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010513, upper bound: 0.0010797
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010506, upper bound: 0.0010796
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011041, upper bound: 0.0010448
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0011048, upper bound: 0.0010431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010819, upper bound: 0.0010957
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010956, upper bound: 0.0010841
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010844, upper bound: 0.0010951
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010962, upper bound: 0.0010815
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010407, upper bound: 0.0010870
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010851, upper bound: 0.0010415
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010409, upper bound: 0.0010860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010864, upper bound: 0.0010414
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010332, upper bound: 0.0010922
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010789, upper bound: 0.0010480
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010823, upper bound: 0.0010774
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010886, upper bound: 0.0010738
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010815, upper bound: 0.0010962
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010840, upper bound: 0.0010956
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010951, upper bound: 0.0010845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 1, lower bound: -0.0010957, upper bound: 0.0010819

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019355, 0.0019420
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007684, 0.0007638
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037410, 0.0036511
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010221, upper bound: 0.0010805
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010192, upper bound: 0.0010803
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019382, 0.0019424
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007686, 0.0007657
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037459, 0.0036925
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010221, upper bound: 0.0010805
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010192, upper bound: 0.0010803
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019362, 0.0019441
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007693, 0.0007638
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036795, 0.0035697
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010258, upper bound: 0.0010908
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010205, upper bound: 0.0010908
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019386, 0.0019413
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007674, 0.0007655
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036408, 0.0036038
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010784
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010785
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019355, 0.0019419
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007683, 0.0007638
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037388, 0.0036506
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010793
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010201, upper bound: 0.0010791
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019382, 0.0019423
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007686, 0.0007657
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037449, 0.0036920
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010793
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010201, upper bound: 0.0010791
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019362, 0.0019440
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007693, 0.0007639
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036784, 0.0035708
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010256, upper bound: 0.0010852
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010256, upper bound: 0.0010852
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019386, 0.0019411
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007672, 0.0007655
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036380, 0.0036036
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010270, upper bound: 0.0010779
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010220, upper bound: 0.0010781
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019372, 0.0019409
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007667, 0.0007641
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036653, 0.0036138
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010381, upper bound: 0.0010692
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010375, upper bound: 0.0010692
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019394, 0.0019382
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007648, 0.0007656
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036283, 0.0036444
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010409, upper bound: 0.0010650
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010401, upper bound: 0.0010649
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019372, 0.0019408
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007666, 0.0007641
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036641, 0.0036149
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010387, upper bound: 0.0010656
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010380, upper bound: 0.0010658
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019395, 0.0019380
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007647, 0.0007657
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036261, 0.0036456
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010416, upper bound: 0.0010600
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010406, upper bound: 0.0010600
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008132
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019334, 0.0019344
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007624, 0.0007617
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036216, 0.0036071
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010342, upper bound: 0.0010672
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010343, upper bound: 0.0010637
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019360, 0.0019349
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007627, 0.0007636
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036274, 0.0036480
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010278, upper bound: 0.0010607
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010297, upper bound: 0.0010564
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019340, 0.0019367
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007636, 0.0007618
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035748, 0.0035383
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010358, upper bound: 0.0010703
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010361, upper bound: 0.0010666
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019365, 0.0019343
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007620, 0.0007635
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035420, 0.0035725
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010391, upper bound: 0.0010664
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010392, upper bound: 0.0010612
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008137, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019383, 0.0019392
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007655, 0.0007649
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036427, 0.0036294
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010613, upper bound: 0.0010374
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010612, upper bound: 0.0010388
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008137, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019383, 0.0019391
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007654, 0.0007649
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036409, 0.0036295
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010348
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010348
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008138
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019406, 0.0019369
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007639, 0.0007665
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036107, 0.0036615
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010792, upper bound: 0.0010212
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010792, upper bound: 0.0010220
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019408, 0.0019367
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007638, 0.0007666
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036081, 0.0036638
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010788, upper bound: 0.0010260
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010788, upper bound: 0.0010266
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019415, 0.0019381
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007652, 0.0007675
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035966, 0.0036436
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010341
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010658, upper bound: 0.0010363
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019423, 0.0019376
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007648, 0.0007681
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035901, 0.0036547
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010682, upper bound: 0.0010365
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010682, upper bound: 0.0010365
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019440, 0.0019358
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007636, 0.0007693
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035652, 0.0036779
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010852, upper bound: 0.0010249
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010852, upper bound: 0.0010249
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019449, 0.0019354
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007632, 0.0007699
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035587, 0.0036902
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010257
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010257
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019446, 0.0019433
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007689, 0.0007698
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036106, 0.0036285
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010374
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010242
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008143
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019473, 0.0019437
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007691, 0.0007717
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036161, 0.0036724
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010374
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010242
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019399, 0.0019401
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007666, 0.0007664
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036244, 0.0036218
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010679, upper bound: 0.0010380
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010679, upper bound: 0.0010380
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019424, 0.0019378
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007649, 0.0007681
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035918, 0.0036554
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010251
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010251
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008137, 0.0008142
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019460, 0.0019392
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007656, 0.0007703
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035998, 0.0036933
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010681, upper bound: 0.0010194
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010204
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008137, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019486, 0.0019396
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007658, 0.0007723
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036044, 0.0037374
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010680, upper bound: 0.0010317
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010808, upper bound: 0.0010194
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019410, 0.0019367
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007638, 0.0007667
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036078, 0.0036669
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010360
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010365
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019435, 0.0019345
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007622, 0.0007685
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035768, 0.0037021
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010908, upper bound: 0.0010237
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010902, upper bound: 0.0010246
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008140
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019426, 0.0019457
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007706, 0.0007684
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037868, 0.0037440
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010214, upper bound: 0.0010805
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010639, upper bound: 0.0010330
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008140
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019429, 0.0019456
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007704, 0.0007686
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037845, 0.0037475
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010576, upper bound: 0.0010766
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010630, upper bound: 0.0010707
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019353, 0.0019420
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007683, 0.0007637
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037402, 0.0036475
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010257, upper bound: 0.0010848
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010264, upper bound: 0.0010775
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019392, 0.0019381
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007656, 0.0007664
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036868, 0.0037019
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010673, upper bound: 0.0010330
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010345
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019359, 0.0019412
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007678, 0.0007641
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037295, 0.0036566
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010210, upper bound: 0.0010809
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010177, upper bound: 0.0010807
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019359, 0.0019411
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007677, 0.0007641
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037288, 0.0036565
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010799
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010186, upper bound: 0.0010797
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019398, 0.0019375
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007652, 0.0007668
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036783, 0.0037099
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010637, upper bound: 0.0010343
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010636, upper bound: 0.0010354
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019399, 0.0019376
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007652, 0.0007669
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036792, 0.0037111
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010672, upper bound: 0.0010342
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010351
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008145, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019495, 0.0019500
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007735, 0.0007732
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037611, 0.0037538
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010693, upper bound: 0.0010653
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010656
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008145, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019499, 0.0019492
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007730, 0.0007735
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037503, 0.0037598
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010353, upper bound: 0.0010723
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010770, upper bound: 0.0010269
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008145, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019488, 0.0019487
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007723, 0.0007723
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037838, 0.0037848
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010329, upper bound: 0.0010644
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010739, upper bound: 0.0010206
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008145, 0.0008145
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019486, 0.0019486
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007722, 0.0007722
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037820, 0.0037825
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010317, upper bound: 0.0010647
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010739, upper bound: 0.0010234
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008137, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019399, 0.0019390
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007658, 0.0007664
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036094, 0.0036218
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010380, upper bound: 0.0010679
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010385, upper bound: 0.0010624
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008138
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019404, 0.0019383
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007653, 0.0007668
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036001, 0.0036291
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010291, upper bound: 0.0010561
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010277, upper bound: 0.0010566
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019475, 0.0019385
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007653, 0.0007716
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036120, 0.0037370
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010852, upper bound: 0.0010256
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010263
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019476, 0.0019385
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007653, 0.0007716
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036121, 0.0037380
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010862, upper bound: 0.0010238
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010859, upper bound: 0.0010247
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008142
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019453, 0.0019461
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007708, 0.0007703
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037923, 0.0037851
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010583, upper bound: 0.0010767
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010633, upper bound: 0.0010705
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019478, 0.0019437
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007692, 0.0007721
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037593, 0.0038197
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010693, upper bound: 0.0010653
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010767, upper bound: 0.0010614
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008142
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019453, 0.0019461
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007708, 0.0007703
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037916, 0.0037853
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010616, upper bound: 0.0010762
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010697
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019477, 0.0019438
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007692, 0.0007720
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037595, 0.0038187
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010704, upper bound: 0.0010628
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010772, upper bound: 0.0010577
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019393, 0.0019401
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007665, 0.0007660
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036304, 0.0036250
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010207, upper bound: 0.0010684
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010229, upper bound: 0.0010612
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008140
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019430, 0.0019364
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007639, 0.0007686
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035788, 0.0036756
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010237
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010666, upper bound: 0.0010220
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019393, 0.0019399
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007664, 0.0007660
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036278, 0.0036248
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010210, upper bound: 0.0010674
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010230, upper bound: 0.0010600
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008140
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019431, 0.0019364
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007640, 0.0007686
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035793, 0.0036770
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010237
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010219
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008132
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019334, 0.0019374
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007645, 0.0007618
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036628, 0.0036115
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010145, upper bound: 0.0010735
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010155, upper bound: 0.0010675
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019371, 0.0019338
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007620, 0.0007644
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036129, 0.0036633
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010561, upper bound: 0.0010302
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010604, upper bound: 0.0010280
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019478, 0.0019461
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007704, 0.0007717
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037188, 0.0037493
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010264, upper bound: 0.0010609
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010672, upper bound: 0.0010185
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008141, 0.0008146
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019503, 0.0019438
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007688, 0.0007734
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036877, 0.0037838
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010769, upper bound: 0.0010618
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010774, upper bound: 0.0010584
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008142
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019462, 0.0019455
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007704, 0.0007709
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037831, 0.0037973
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010177, upper bound: 0.0010807
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010636, upper bound: 0.0010354
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008142
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019462, 0.0019455
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007704, 0.0007709
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037837, 0.0037973
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010767
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010653, upper bound: 0.0010699
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019484, 0.0019430
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007686, 0.0007725
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037484, 0.0038285
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010321, upper bound: 0.0010674
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010793, upper bound: 0.0010234
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008144
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019485, 0.0019430
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007686, 0.0007726
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037490, 0.0038293
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010322, upper bound: 0.0010640
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010805, upper bound: 0.0010221
time: 0.95 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010221, upper bound: 0.0010805
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010192, upper bound: 0.0010803
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010221, upper bound: 0.0010805
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010192, upper bound: 0.0010803
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010258, upper bound: 0.0010908
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010205, upper bound: 0.0010908
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010784
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010785
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010793
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010201, upper bound: 0.0010791
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010793
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010201, upper bound: 0.0010791
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010256, upper bound: 0.0010852
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010256, upper bound: 0.0010852
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010270, upper bound: 0.0010779
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010220, upper bound: 0.0010781
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010381, upper bound: 0.0010692
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010375, upper bound: 0.0010692
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010409, upper bound: 0.0010650
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010401, upper bound: 0.0010649
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010387, upper bound: 0.0010656
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010380, upper bound: 0.0010658
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010416, upper bound: 0.0010600
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010406, upper bound: 0.0010600
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010342, upper bound: 0.0010672
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010343, upper bound: 0.0010637
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010278, upper bound: 0.0010607
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010297, upper bound: 0.0010564
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010358, upper bound: 0.0010703
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010361, upper bound: 0.0010666
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010391, upper bound: 0.0010664
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010392, upper bound: 0.0010612
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010613, upper bound: 0.0010374
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010612, upper bound: 0.0010388
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010792, upper bound: 0.0010212
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010792, upper bound: 0.0010220
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010788, upper bound: 0.0010260
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010788, upper bound: 0.0010266
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010341
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010658, upper bound: 0.0010363
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010682, upper bound: 0.0010365
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010682, upper bound: 0.0010365
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010852, upper bound: 0.0010249
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010852, upper bound: 0.0010249
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010257
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010257
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010242
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010242
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010679, upper bound: 0.0010380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010679, upper bound: 0.0010380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010790, upper bound: 0.0010251
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010681, upper bound: 0.0010194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010204
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010680, upper bound: 0.0010317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010808, upper bound: 0.0010194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010365
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010908, upper bound: 0.0010237
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010902, upper bound: 0.0010246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010214, upper bound: 0.0010805
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010639, upper bound: 0.0010330
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010576, upper bound: 0.0010766
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010630, upper bound: 0.0010707
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010257, upper bound: 0.0010848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010264, upper bound: 0.0010775
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010673, upper bound: 0.0010330
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010345
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010210, upper bound: 0.0010809
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010177, upper bound: 0.0010807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010799
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010186, upper bound: 0.0010797
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010637, upper bound: 0.0010343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010636, upper bound: 0.0010354
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010672, upper bound: 0.0010342
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010351
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010693, upper bound: 0.0010653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010656
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010353, upper bound: 0.0010723
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010770, upper bound: 0.0010269
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010329, upper bound: 0.0010644
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010739, upper bound: 0.0010206
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010317, upper bound: 0.0010647
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010739, upper bound: 0.0010234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010380, upper bound: 0.0010679
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010385, upper bound: 0.0010624
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010291, upper bound: 0.0010561
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010277, upper bound: 0.0010566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010852, upper bound: 0.0010256
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010263
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010862, upper bound: 0.0010238
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010859, upper bound: 0.0010247
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010583, upper bound: 0.0010767
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010633, upper bound: 0.0010705
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010693, upper bound: 0.0010653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010767, upper bound: 0.0010614
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010616, upper bound: 0.0010762
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010697
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010704, upper bound: 0.0010628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010772, upper bound: 0.0010577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010207, upper bound: 0.0010684
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010229, upper bound: 0.0010612
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010237
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010666, upper bound: 0.0010220
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010210, upper bound: 0.0010674
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010230, upper bound: 0.0010600
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010237
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010219
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010145, upper bound: 0.0010735
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010155, upper bound: 0.0010675
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010561, upper bound: 0.0010302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010604, upper bound: 0.0010280
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010264, upper bound: 0.0010609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010672, upper bound: 0.0010185
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010769, upper bound: 0.0010618
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010774, upper bound: 0.0010584
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010177, upper bound: 0.0010807
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010636, upper bound: 0.0010354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010653, upper bound: 0.0010699
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010321, upper bound: 0.0010674
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010793, upper bound: 0.0010234
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010322, upper bound: 0.0010640
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 1, lower bound: -0.0010805, upper bound: 0.0010221

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008129
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019284, 0.0019350
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007628, 0.0007582
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036265, 0.0035348
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010037, upper bound: 0.0010619
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010045, upper bound: 0.0010548
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008129
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019285, 0.0019345
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007624, 0.0007583
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036189, 0.0035367
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010010, upper bound: 0.0010618
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010549
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019310, 0.0019354
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007631, 0.0007601
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036314, 0.0035759
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010037, upper bound: 0.0010619
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010045, upper bound: 0.0010548
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019312, 0.0019349
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007628, 0.0007602
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036257, 0.0035778
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010010, upper bound: 0.0010618
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010549
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008129
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019294, 0.0019372
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007640, 0.0007586
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035794, 0.0034717
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010030, upper bound: 0.0010623
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010030, upper bound: 0.0010623
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008129
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019293, 0.0019368
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007637, 0.0007585
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035736, 0.0034696
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0009992, upper bound: 0.0010621
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0009992, upper bound: 0.0010621
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019360, 0.0019381
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007651, 0.0007636
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035923, 0.0035626
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010040, upper bound: 0.0010548
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010002, upper bound: 0.0010549
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019386, 0.0019386
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007655, 0.0007655
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035996, 0.0036038
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010040, upper bound: 0.0010548
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010002, upper bound: 0.0010549
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008128
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019283, 0.0019349
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007627, 0.0007582
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036244, 0.0035342
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010052, upper bound: 0.0010610
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010058, upper bound: 0.0010535
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008129
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019285, 0.0019344
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007624, 0.0007583
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036176, 0.0035362
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010608
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010024, upper bound: 0.0010535
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019310, 0.0019353
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007630, 0.0007601
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036304, 0.0035754
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010052, upper bound: 0.0010610
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010058, upper bound: 0.0010535
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019311, 0.0019348
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007626, 0.0007602
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036230, 0.0035773
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010608
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010024, upper bound: 0.0010535
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008138, 0.0008132
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019336, 0.0019410
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007671, 0.0007620
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036322, 0.0035296
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010046, upper bound: 0.0010614
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010004, upper bound: 0.0010612
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019362, 0.0019414
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007674, 0.0007639
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036373, 0.0035708
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010046, upper bound: 0.0010614
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010004, upper bound: 0.0010612
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019316, 0.0019342
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007619, 0.0007601
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035379, 0.0035017
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010053, upper bound: 0.0010535
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010053, upper bound: 0.0010535
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019317, 0.0019341
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007619, 0.0007602
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035370, 0.0035035
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010012, upper bound: 0.0010535
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010012, upper bound: 0.0010535
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019313, 0.0019355
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007628, 0.0007599
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035550, 0.0034971
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010154, upper bound: 0.0010484
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010154, upper bound: 0.0010484
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019317, 0.0019348
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007623, 0.0007602
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035458, 0.0035035
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010484
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010484
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019335, 0.0019328
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007609, 0.0007614
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035180, 0.0035278
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010173, upper bound: 0.0010439
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010173, upper bound: 0.0010439
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019339, 0.0019322
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007605, 0.0007617
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035095, 0.0035341
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010165, upper bound: 0.0010438
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010165, upper bound: 0.0010438
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019314, 0.0019354
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007627, 0.0007600
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035538, 0.0034990
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010157, upper bound: 0.0010450
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010157, upper bound: 0.0010450
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019318, 0.0019346
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007622, 0.0007602
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035435, 0.0035045
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010149, upper bound: 0.0010453
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010149, upper bound: 0.0010453
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019334, 0.0019326
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007608, 0.0007614
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035158, 0.0035273
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010175, upper bound: 0.0010392
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010175, upper bound: 0.0010392
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019340, 0.0019320
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007604, 0.0007618
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035073, 0.0035353
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010168, upper bound: 0.0010393
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010168, upper bound: 0.0010393
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019309, 0.0019320
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007607, 0.0007600
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035854, 0.0035699
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010488
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010445
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008130
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019308, 0.0019320
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007607, 0.0007599
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035845, 0.0035690
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010147, upper bound: 0.0010452
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010397
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019336, 0.0019348
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007623, 0.0007615
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035416, 0.0035325
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010488
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010147, upper bound: 0.0010452
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019360, 0.0019323
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007606, 0.0007631
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035079, 0.0035650
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010445
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010397
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019316, 0.0019345
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007621, 0.0007601
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035412, 0.0035015
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010488
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010488
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019316, 0.0019342
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007619, 0.0007601
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035380, 0.0035024
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010455
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010455
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019341, 0.0019321
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007605, 0.0007618
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035092, 0.0035357
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010445
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010445
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019341, 0.0019318
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007603, 0.0007619
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035052, 0.0035365
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010399
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010399
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008132
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019322, 0.0019338
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007617, 0.0007605
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035324, 0.0035096
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010401, upper bound: 0.0010143
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010401, upper bound: 0.0010143
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008132
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019329, 0.0019332
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007612, 0.0007610
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035241, 0.0035191
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010400, upper bound: 0.0010152
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010400, upper bound: 0.0010152
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019356, 0.0019361
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007632, 0.0007629
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035927, 0.0035870
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010394, upper bound: 0.0010157
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010393, upper bound: 0.0010162
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019383, 0.0019365
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007635, 0.0007649
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035984, 0.0036295
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010394, upper bound: 0.0010157
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010393, upper bound: 0.0010162
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019345, 0.0019315
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007600, 0.0007621
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035003, 0.0035415
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010008
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010008
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008130, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019352, 0.0019308
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007596, 0.0007626
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034913, 0.0035512
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010020
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010020
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019345, 0.0019313
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007599, 0.0007621
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034978, 0.0035414
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010046
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010046
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008130, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019353, 0.0019306
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007594, 0.0007627
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034884, 0.0035535
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010052
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010052
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019344, 0.0019312
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007598, 0.0007621
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034965, 0.0035408
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010456, upper bound: 0.0010119
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010456, upper bound: 0.0010120
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008133
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019346, 0.0019314
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007600, 0.0007622
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034989, 0.0035435
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010136
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010136
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019396, 0.0019345
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007626, 0.0007662
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035427, 0.0036135
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010131
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010450, upper bound: 0.0010142
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019423, 0.0019350
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007629, 0.0007681
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035490, 0.0036547
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010131
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010450, upper bound: 0.0010142
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019413, 0.0019327
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007614, 0.0007674
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035181, 0.0036367
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010613, upper bound: 0.0009999
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010038
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019440, 0.0019332
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007617, 0.0007693
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035240, 0.0036779
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010613, upper bound: 0.0009999
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010038
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019422, 0.0019321
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007610, 0.0007680
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035095, 0.0036491
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010608, upper bound: 0.0010010
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010045
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008141
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019449, 0.0019327
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007614, 0.0007699
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035175, 0.0036902
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010608, upper bound: 0.0010010
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010045
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019366, 0.0019377
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007648, 0.0007641
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035866, 0.0035718
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010142
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010155
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019389, 0.0019353
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007632, 0.0007657
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035539, 0.0036032
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0009999
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010033
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019393, 0.0019380
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007650, 0.0007660
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035906, 0.0036129
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010142
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010155
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019416, 0.0019357
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007635, 0.0007676
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035594, 0.0036443
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0009999
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010033
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008135, 0.0008135
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019373, 0.0019371
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007645, 0.0007645
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035790, 0.0035806
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010152
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010159
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008136, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019399, 0.0019374
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007647, 0.0007664
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035832, 0.0036218
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010152
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010159
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008137
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019397, 0.0019347
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007628, 0.0007662
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035458, 0.0036142
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0010013
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010038
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008139
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019424, 0.0019351
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007630, 0.0007681
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035506, 0.0036554
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0010013
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010038
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008138
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019398, 0.0019337
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007619, 0.0007661
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034871, 0.0035712
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010119
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010621, upper bound: 0.0009988
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008132, 0.0008138
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019404, 0.0019331
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007615, 0.0007666
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034794, 0.0035806
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010131
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010618, upper bound: 0.0010003
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008133, 0.0008138
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019407, 0.0019339
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007618, 0.0007666
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035637, 0.0036633
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010119
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010131
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008140
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019432, 0.0019316
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007602, 0.0007683
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035318, 0.0036972
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010621, upper bound: 0.0009988
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010618, upper bound: 0.0010003
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019348, 0.0019313
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007599, 0.0007623
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034975, 0.0035461
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010134
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010134
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019356, 0.0019309
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007596, 0.0007629
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034919, 0.0035566
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010140
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010140
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008129, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019372, 0.0019291
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007583, 0.0007640
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034665, 0.0035791
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010623, upper bound: 0.0010024
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010623, upper bound: 0.0010024
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008129, 0.0008136
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019381, 0.0019285
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007580, 0.0007646
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0034591, 0.0035917
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010619, upper bound: 0.0010031
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010619, upper bound: 0.0010031
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008134, 0.0008128
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019280, 0.0019350
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007628, 0.0007579
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036267, 0.0035298
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010031, upper bound: 0.0010619
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010038, upper bound: 0.0010552
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008131, 0.0008131
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019316, 0.0019311
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007601, 0.0007605
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0035726, 0.0035797
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010399, upper bound: 0.0010153
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010455, upper bound: 0.0010132
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008142, 0.0008138
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019403, 0.0019455
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007700, 0.0007664
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0037089, 0.0036374
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010003, upper bound: 0.0010618
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010393, upper bound: 0.0010168
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008140, 0.0008140
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019428, 0.0019433
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007685, 0.0007681
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036784, 0.0036719
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010013, upper bound: 0.0010553
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010149
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008139, 0.0008132
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019327, 0.0019418
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007677, 0.0007614
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036430, 0.0035175
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010045, upper bound: 0.0010610
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010010, upper bound: 0.0010608
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0006058, 0.0014308, 0.0006058, 0.0014308, -0.0008137, 0.0008134
1: 0.9928539, 0.9949342, 0.9928539, 0.9949342, -0.0019351, 0.0019391
2: -0.0071486, -0.0045147, -0.0071486, -0.0045147, -0.0026339, 0.0026339
3: 0.0034449, 0.0043078, 0.0034449, 0.0043078, -0.0007658, 0.0007630
4: 0.0023313, 0.0040669, 0.0023313, 0.0040669, -0.0017356, 0.0017356
5: 0.0051756, 0.0071975, 0.0051756, 0.0071975, -0.0020219, 0.0020219
6: -0.0016024, -0.0006666, -0.0016024, -0.0006666, -0.0009358, 0.0009358
7: -0.0087947, -0.0073388, -0.0087947, -0.0073388, -0.0014559, 0.0014559
8: 0.0030395, 0.0076440, 0.0030395, 0.0076440, -0.0036060, 0.0035503
9: -0.0047604, -0.0021201, -0.0047604, -0.0021201, -0.0026403, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010052, upper bound: 0.0010539
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010020, upper bound: 0.0010539
time: 1.08 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010037, upper bound: 0.0010619
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010045, upper bound: 0.0010548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010010, upper bound: 0.0010618
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010549
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010037, upper bound: 0.0010619
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010045, upper bound: 0.0010548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010010, upper bound: 0.0010618
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010549
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010030, upper bound: 0.0010623
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010030, upper bound: 0.0010623
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0009992, upper bound: 0.0010621
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0009992, upper bound: 0.0010621
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010040, upper bound: 0.0010548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010002, upper bound: 0.0010549
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010040, upper bound: 0.0010548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010002, upper bound: 0.0010549
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010052, upper bound: 0.0010610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010058, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010608
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010024, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010052, upper bound: 0.0010610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010058, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010017, upper bound: 0.0010608
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010024, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010046, upper bound: 0.0010614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010004, upper bound: 0.0010612
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010046, upper bound: 0.0010614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010004, upper bound: 0.0010612
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010053, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010053, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010012, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010012, upper bound: 0.0010535
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010154, upper bound: 0.0010484
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010154, upper bound: 0.0010484
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010484
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010484
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010173, upper bound: 0.0010439
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010173, upper bound: 0.0010439
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010165, upper bound: 0.0010438
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010165, upper bound: 0.0010438
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010157, upper bound: 0.0010450
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010157, upper bound: 0.0010450
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010149, upper bound: 0.0010453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010149, upper bound: 0.0010453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010175, upper bound: 0.0010392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010175, upper bound: 0.0010392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010168, upper bound: 0.0010393
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010168, upper bound: 0.0010393
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010488
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010445
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010147, upper bound: 0.0010452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010397
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010146, upper bound: 0.0010488
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010147, upper bound: 0.0010452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010445
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010166, upper bound: 0.0010397
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010488
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010488
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010455
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010132, upper bound: 0.0010455
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010445
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010445
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010399
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010153, upper bound: 0.0010399
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010401, upper bound: 0.0010143
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010401, upper bound: 0.0010143
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010400, upper bound: 0.0010152
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010400, upper bound: 0.0010152
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010394, upper bound: 0.0010157
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010393, upper bound: 0.0010162
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010394, upper bound: 0.0010157
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010393, upper bound: 0.0010162
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010008
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010008
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010020
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010020
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010046
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010046
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010052
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010539, upper bound: 0.0010052
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010456, upper bound: 0.0010119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010456, upper bound: 0.0010120
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010136
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010136
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010450, upper bound: 0.0010142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010450, upper bound: 0.0010142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010613, upper bound: 0.0009999
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010038
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010613, upper bound: 0.0009999
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010038
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010608, upper bound: 0.0010010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010045
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010608, upper bound: 0.0010010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010045
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010155
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0009999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010033
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010155
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0009999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010033
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010152
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010159
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010446, upper bound: 0.0010152
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010439, upper bound: 0.0010159
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0010013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010038
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010553, upper bound: 0.0010013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010552, upper bound: 0.0010038
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010119
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010621, upper bound: 0.0009988
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010618, upper bound: 0.0010003
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010119
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010490, upper bound: 0.0010131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010621, upper bound: 0.0009988
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010618, upper bound: 0.0010003
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010134
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010134
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010140
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010485, upper bound: 0.0010140
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010623, upper bound: 0.0010024
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010623, upper bound: 0.0010024
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010619, upper bound: 0.0010031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010619, upper bound: 0.0010031
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010031, upper bound: 0.0010619
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010038, upper bound: 0.0010552
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010399, upper bound: 0.0010153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010455, upper bound: 0.0010132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010003, upper bound: 0.0010618
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010393, upper bound: 0.0010168
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010013, upper bound: 0.0010553
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010453, upper bound: 0.0010149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010045, upper bound: 0.0010610
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010010, upper bound: 0.0010608
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010052, upper bound: 0.0010539
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0010020, upper bound: 0.0010539
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010673, upper bound: 0.0010330
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010345
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010210, upper bound: 0.0010809
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010177, upper bound: 0.0010807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010799
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010186, upper bound: 0.0010797
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010637, upper bound: 0.0010343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010636, upper bound: 0.0010354
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010672, upper bound: 0.0010342
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010668, upper bound: 0.0010351
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010693, upper bound: 0.0010653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010656
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010353, upper bound: 0.0010723
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010770, upper bound: 0.0010269
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010329, upper bound: 0.0010644
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010739, upper bound: 0.0010206
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010317, upper bound: 0.0010647
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010739, upper bound: 0.0010234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010380, upper bound: 0.0010679
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010385, upper bound: 0.0010624
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010291, upper bound: 0.0010561
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010277, upper bound: 0.0010566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010852, upper bound: 0.0010256
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010263
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010862, upper bound: 0.0010238
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010859, upper bound: 0.0010247
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010583, upper bound: 0.0010767
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010633, upper bound: 0.0010705
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010693, upper bound: 0.0010653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010767, upper bound: 0.0010614
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010616, upper bound: 0.0010762
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010697
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010704, upper bound: 0.0010628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010772, upper bound: 0.0010577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010207, upper bound: 0.0010684
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010229, upper bound: 0.0010612
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010237
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010666, upper bound: 0.0010220
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010210, upper bound: 0.0010674
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010230, upper bound: 0.0010600
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010610, upper bound: 0.0010237
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010678, upper bound: 0.0010219
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010145, upper bound: 0.0010735
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010155, upper bound: 0.0010675
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010561, upper bound: 0.0010302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010604, upper bound: 0.0010280
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010264, upper bound: 0.0010609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010672, upper bound: 0.0010185
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010769, upper bound: 0.0010618
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010774, upper bound: 0.0010584
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010177, upper bound: 0.0010807
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010636, upper bound: 0.0010354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010614, upper bound: 0.0010767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010653, upper bound: 0.0010699
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010321, upper bound: 0.0010674
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010793, upper bound: 0.0010234
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010322, upper bound: 0.0010640
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0010805, upper bound: 0.0010221

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.59 + 598.48 = 602.07 seconds
