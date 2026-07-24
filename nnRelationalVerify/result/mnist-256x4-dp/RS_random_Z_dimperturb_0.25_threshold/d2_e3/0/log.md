## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00886005


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009595, 0.0009595)
1: (-0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0027409, 0.0027409)
2: (0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0039059, 0.0039059)
3: (-0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0028472, 0.0028472)
4: (-0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0033469, 0.0033469)
5: (0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0028336, 0.0028336)
6: (0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758)
7: (-0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0053876, 0.0053876)
8: (0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0184414, 0.0184414)
9: (0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0047062, 0.0047062)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.39 = 2.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0120578, upper bound: 0.0120578

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119427, upper bound: 0.0119466
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0119466, upper bound: 0.0119427
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 8, lower bound: -0.0119427, upper bound: 0.0119466
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 8, lower bound: -0.0119466, upper bound: 0.0119427

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009506, 0.0009510
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026970, 0.0026987
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0038412, 0.0038386
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027980, 0.0027961
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0033068, 0.0033051
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027843, 0.0027824
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0052820, 0.0052862
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0181256, 0.0181375
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0046185, 0.0046150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118562, upper bound: 0.0118203
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118203, upper bound: 0.0118591
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009510, 0.0009506
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026987, 0.0026970
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0038386, 0.0038412
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027961, 0.0027980
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0033051, 0.0033068
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027824, 0.0027843
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0052862, 0.0052820
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0181375, 0.0181256
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0046150, 0.0046185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116554, upper bound: 0.0116485
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116530, upper bound: 0.0116536
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 8, lower bound: -0.0118562, upper bound: 0.0118203
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 8, lower bound: -0.0118203, upper bound: 0.0118591
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 8, lower bound: -0.0116554, upper bound: 0.0116485
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 8, lower bound: -0.0116530, upper bound: 0.0116536

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009445, 0.0009436
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026667, 0.0026624
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037868, 0.0037932
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027571, 0.0027619
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032691, 0.0032736
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027435, 0.0027483
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0052080, 0.0051976
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0179135, 0.0178837
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0045439, 0.0045527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115726, upper bound: 0.0115249
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115647, upper bound: 0.0115257
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009433, 0.0009449
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026607, 0.0026683
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037956, 0.0037842
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027637, 0.0027552
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032752, 0.0032674
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027501, 0.0027416
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0051934, 0.0052119
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0178717, 0.0179247
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0045560, 0.0045404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116868, upper bound: 0.0117976
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117588, upper bound: 0.0117285
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009427, 0.0009404
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026598, 0.0026489
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037666, 0.0037828
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027420, 0.0027541
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032551, 0.0032664
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027284, 0.0027406
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0051912, 0.0051647
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0178652, 0.0177895
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0045162, 0.0045385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115252, upper bound: 0.0114595
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114600, upper bound: 0.0115184
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009408, 0.0009422
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026506, 0.0026575
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037795, 0.0037692
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027516, 0.0027439
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032640, 0.0032569
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027380, 0.0027303
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0051689, 0.0051856
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0178014, 0.0178494
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0045338, 0.0045197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112333, upper bound: 0.0112326
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112333, upper bound: 0.0112326
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0115726, upper bound: 0.0115249
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0115647, upper bound: 0.0115257
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0116868, upper bound: 0.0117976
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0117588, upper bound: 0.0117285
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0115252, upper bound: 0.0114595
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0114600, upper bound: 0.0115184
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0112333, upper bound: 0.0112326
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 8, lower bound: -0.0112333, upper bound: 0.0112326

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009361, 0.0009334
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026274, 0.0026145
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037150, 0.0037344
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027031, 0.0027178
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032193, 0.0032328
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026897, 0.0027042
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0051123, 0.0050806
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0176393, 0.0175486
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044454, 0.0044721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114446, upper bound: 0.0113527
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113739, upper bound: 0.0113946
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009343, 0.0009353
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026187, 0.0026236
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037287, 0.0037214
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027134, 0.0027079
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032288, 0.0032238
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026999, 0.0026944
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050910, 0.0051029
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175784, 0.0176123
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044642, 0.0044542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101736, upper bound: 0.0112274
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112646, upper bound: 0.0101605
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009188, 0.0009213
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025114, 0.0025232
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035739, 0.0035562
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025963, 0.0025830
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031433, 0.0031311
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025829, 0.0025696
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0047932, 0.0048220
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0168104, 0.0168930
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0042413, 0.0042171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115547, upper bound: 0.0115993
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115230, upper bound: 0.0116646
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009196, 0.0009204
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025152, 0.0025190
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035676, 0.0035620
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025916, 0.0025873
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031389, 0.0031351
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025782, 0.0025740
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0048026, 0.0048117
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0168373, 0.0168634
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0042326, 0.0042249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103566, upper bound: 0.0114213
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114827, upper bound: 0.0104204
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009403, 0.0009369
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026493, 0.0026332
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037430, 0.0037671
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027242, 0.0027423
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032387, 0.0032555
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027107, 0.0027288
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0051655, 0.0051262
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0177918, 0.0176792
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044838, 0.0045169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114442, upper bound: 0.0113553
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113954, upper bound: 0.0113726
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009392, 0.0009404
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026440, 0.0026489
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037666, 0.0037592
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027420, 0.0027364
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032551, 0.0032500
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027284, 0.0027228
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0051526, 0.0051647
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0177548, 0.0177895
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0045162, 0.0045060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113737, upper bound: 0.0113952
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113527, upper bound: 0.0114354
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009345, 0.0009386
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026259, 0.0026453
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037598, 0.0037308
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027371, 0.0027152
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032521, 0.0032319
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027235, 0.0027017
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050964, 0.0051437
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0176222, 0.0177579
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0045056, 0.0044657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105379, upper bound: 0.0105395
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105407, upper bound: 0.0105345
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009408, 0.0009359
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026506, 0.0026328
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037410, 0.0037692
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027229, 0.0027439
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032390, 0.0032569
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0027094, 0.0027303
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0051689, 0.0051131
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0178014, 0.0176701
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044798, 0.0045197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110802, upper bound: 0.0111818
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111824, upper bound: 0.0110737
time: 0.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0114446, upper bound: 0.0113527
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0113739, upper bound: 0.0113946
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0101736, upper bound: 0.0112274
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0112646, upper bound: 0.0101605
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0115547, upper bound: 0.0115993
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0115230, upper bound: 0.0116646
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0103566, upper bound: 0.0114213
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0114827, upper bound: 0.0104204
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0114442, upper bound: 0.0113553
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0113954, upper bound: 0.0113726
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0113737, upper bound: 0.0113952
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0113527, upper bound: 0.0114354
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0105379, upper bound: 0.0105395
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0105407, upper bound: 0.0105345
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0110802, upper bound: 0.0111818
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 8, lower bound: -0.0111824, upper bound: 0.0110737

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009338, 0.0009298
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026169, 0.0025984
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036909, 0.0037187
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026850, 0.0027059
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032026, 0.0032219
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026716, 0.0026924
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050866, 0.0050414
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175657, 0.0174361
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044124, 0.0044505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110466, upper bound: 0.0110087
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110628, upper bound: 0.0109567
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009326, 0.0009334
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026114, 0.0026145
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037150, 0.0037104
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027031, 0.0026996
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032193, 0.0032161
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026897, 0.0026862
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050731, 0.0050806
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175268, 0.0175486
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044454, 0.0044390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103017, upper bound: 0.0102815
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103017, upper bound: 0.0102815
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008678, 0.0008929
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024054, 0.0025229
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035495, 0.0033735
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025651, 0.0024328
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031700, 0.0030480
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025505, 0.0024185
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0043413, 0.0046280
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0159864, 0.0168078
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041157, 0.0038743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100934, upper bound: 0.0111741
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101189, upper bound: 0.0110973
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008939, 0.0008688
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025275, 0.0024102
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033808, 0.0035564
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024382, 0.0025703
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030530, 0.0031748
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024239, 0.0025558
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046394, 0.0043532
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0168404, 0.0160204
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038843, 0.0041253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111095, upper bound: 0.0099718
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110540, upper bound: 0.0100214
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009162, 0.0009179
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024974, 0.0025052
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035470, 0.0035353
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025760, 0.0025673
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031246, 0.0031165
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025627, 0.0025539
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0047591, 0.0047781
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167126, 0.0167671
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0042043, 0.0041883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108190, upper bound: 0.0109159
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108190, upper bound: 0.0109159
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009154, 0.0009213
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024934, 0.0025232
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035739, 0.0035293
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025963, 0.0025627
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031433, 0.0031124
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025829, 0.0025494
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0047493, 0.0048220
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0166846, 0.0168930
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0042413, 0.0041801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111487, upper bound: 0.0113153
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111926, upper bound: 0.0113021
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008493, 0.0008754
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023088, 0.0024311
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033993, 0.0032161
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024457, 0.0023079
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030992, 0.0029721
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024307, 0.0022932
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040051, 0.0043036
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0152644, 0.0161197
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038438, 0.0035924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100727, upper bound: 0.0111185
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100727, upper bound: 0.0110740
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008770, 0.0008501
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024386, 0.0023126
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032217, 0.0034104
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023122, 0.0024541
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029760, 0.0031069
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022974, 0.0024391
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0043217, 0.0040142
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0161716, 0.0152905
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036001, 0.0038590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111905, upper bound: 0.0101348
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111972, upper bound: 0.0101348
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009343, 0.0009295
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026195, 0.0025967
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036884, 0.0037225
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026831, 0.0027088
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032009, 0.0032245
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026697, 0.0026953
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050928, 0.0050372
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175835, 0.0174242
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044089, 0.0044557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113378, upper bound: 0.0113026
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113932, upper bound: 0.0112423
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009329, 0.0009307
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026128, 0.0026027
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036973, 0.0037125
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026898, 0.0027013
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032071, 0.0032176
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026764, 0.0026878
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050766, 0.0050518
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175369, 0.0174659
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044211, 0.0044420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009331, 0.0009330
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026138, 0.0026128
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037125, 0.0037141
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027012, 0.0027024
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032176, 0.0032187
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026877, 0.0026889
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050791, 0.0050765
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175441, 0.0175367
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044419, 0.0044441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103125, upper bound: 0.0102607
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103125, upper bound: 0.0102607
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009318, 0.0009343
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026075, 0.0026187
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037214, 0.0037046
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0027079, 0.0026953
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032238, 0.0032121
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026944, 0.0026818
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050637, 0.0050910
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0174999, 0.0175784
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044542, 0.0044311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009218, 0.0009262
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025614, 0.0025821
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036618, 0.0036308
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026620, 0.0026387
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031906, 0.0031691
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026486, 0.0026253
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0049474, 0.0049978
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0171595, 0.0173042
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043795, 0.0043370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092187, upper bound: 0.0101393
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101435, upper bound: 0.0091849
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009220, 0.0009259
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025624, 0.0025808
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036599, 0.0036323
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026606, 0.0026398
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031892, 0.0031701
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026471, 0.0026264
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0049497, 0.0049947
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0171663, 0.0172952
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043769, 0.0043390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104363, upper bound: 0.0103954
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103995, upper bound: 0.0104311
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009164, 0.0009114
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024963, 0.0024771
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035040, 0.0035337
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025432, 0.0025661
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030973, 0.0031154
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025299, 0.0025527
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0047565, 0.0047031
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167053, 0.0165674
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041404, 0.0041862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109858, upper bound: 0.0110521
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109405, upper bound: 0.0110834
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009177, 0.0009111
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025023, 0.0024754
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035015, 0.0035427
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025414, 0.0025729
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030956, 0.0031217
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025280, 0.0025595
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0047712, 0.0046991
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167473, 0.0165559
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041370, 0.0041985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098236, upper bound: 0.0107035
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108275, upper bound: 0.0097840
time: 0.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0110466, upper bound: 0.0110087
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0110628, upper bound: 0.0109567
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0103017, upper bound: 0.0102815
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0103017, upper bound: 0.0102815
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0100934, upper bound: 0.0111741
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0101189, upper bound: 0.0110973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0111095, upper bound: 0.0099718
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0110540, upper bound: 0.0100214
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0108190, upper bound: 0.0109159
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0108190, upper bound: 0.0109159
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0111487, upper bound: 0.0113153
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0111926, upper bound: 0.0113021
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0100727, upper bound: 0.0111185
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0100727, upper bound: 0.0110740
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0111905, upper bound: 0.0101348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0111972, upper bound: 0.0101348
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0113378, upper bound: 0.0113026
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0113932, upper bound: 0.0112423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0103125, upper bound: 0.0102607
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0103125, upper bound: 0.0102607
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0102815, upper bound: 0.0103017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0092187, upper bound: 0.0101393
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0101435, upper bound: 0.0091849
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0104363, upper bound: 0.0103954
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0103995, upper bound: 0.0104311
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0109858, upper bound: 0.0110521
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0109405, upper bound: 0.0110834
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0098236, upper bound: 0.0107035
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 8, lower bound: -0.0108275, upper bound: 0.0097840

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009209, 0.0009171
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025523, 0.0025348
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035919, 0.0036182
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026092, 0.0026290
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031410, 0.0031593
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025960, 0.0026157
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0049366, 0.0048937
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0171012, 0.0169784
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0042831, 0.0043191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0085424, upper bound: 0.0085288
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0085424, upper bound: 0.0085288
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009211, 0.0009169
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025535, 0.0025338
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035905, 0.0036200
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026081, 0.0026303
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031400, 0.0031605
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025949, 0.0026170
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0049395, 0.0048914
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0171095, 0.0169716
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0042811, 0.0043216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109548, upper bound: 0.0109088
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110176, upper bound: 0.0108608
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009263, 0.0009304
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025866, 0.0026053
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036998, 0.0036719
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026919, 0.0026710
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032105, 0.0031911
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026784, 0.0026575
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050005, 0.0050459
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0173475, 0.0174777
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044233, 0.0043850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090195, upper bound: 0.0097669
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009326, 0.0009271
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026114, 0.0025897
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036766, 0.0037104
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026745, 0.0026996
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031944, 0.0032161
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026610, 0.0026862
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050731, 0.0050081
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175268, 0.0173693
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043914, 0.0044390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090195, upper bound: 0.0097669
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008396, 0.0008669
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022581, 0.0023858
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033314, 0.0031401
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023947, 0.0022508
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030521, 0.0029194
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023798, 0.0022362
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038813, 0.0041931
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149097, 0.0158029
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037507, 0.0034882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096804, upper bound: 0.0106855
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096804, upper bound: 0.0106855
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008402, 0.0008647
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022611, 0.0023756
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033160, 0.0031446
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023831, 0.0022542
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030414, 0.0029225
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023682, 0.0022395
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038886, 0.0041680
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149306, 0.0157311
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037296, 0.0034943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097696, upper bound: 0.0107738
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098005, upper bound: 0.0107200
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008913, 0.0008653
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025161, 0.0023942
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033567, 0.0035393
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024201, 0.0025574
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030363, 0.0031629
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024059, 0.0025429
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046114, 0.0043139
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167603, 0.0159079
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038512, 0.0041018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106080, upper bound: 0.0096112
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106248, upper bound: 0.0095926
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008904, 0.0008688
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025114, 0.0024102
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033808, 0.0035324
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024382, 0.0025522
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030530, 0.0031581
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024239, 0.0025377
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046001, 0.0043532
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167279, 0.0160204
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038843, 0.0040922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009095, 0.0009135
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024696, 0.0024886
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035212, 0.0034928
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025562, 0.0025348
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031092, 0.0030895
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025428, 0.0025215
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046849, 0.0047312
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0165152, 0.0166477
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041640, 0.0041250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009162, 0.0009111
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024974, 0.0024774
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035044, 0.0035353
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025436, 0.0025673
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030976, 0.0031165
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025303, 0.0025539
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0047591, 0.0047039
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167126, 0.0165697
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041410, 0.0041883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009021, 0.0009088
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024328, 0.0024660
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034848, 0.0034351
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025263, 0.0024889
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030837, 0.0030492
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025127, 0.0024754
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045964, 0.0046774
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162518, 0.0164840
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041103, 0.0040420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097959, upper bound: 0.0109401
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107814, upper bound: 0.0099526
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009024, 0.0009080
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024344, 0.0024626
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034797, 0.0034375
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025225, 0.0024907
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030801, 0.0030508
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025089, 0.0024772
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046003, 0.0046692
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162629, 0.0164602
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041033, 0.0040453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109035, upper bound: 0.0110025
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109022, upper bound: 0.0110042
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008428, 0.0008742
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022848, 0.0024316
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033999, 0.0031799
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024461, 0.0022807
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030996, 0.0029470
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024310, 0.0022659
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039367, 0.0042951
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0150947, 0.0161217
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038394, 0.0035376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095790, upper bound: 0.0104017
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095790, upper bound: 0.0104017
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008493, 0.0008690
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023088, 0.0024071
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033631, 0.0032161
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024184, 0.0023079
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030741, 0.0029721
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024034, 0.0022932
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040051, 0.0042352
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0152644, 0.0159500
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037890, 0.0035924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096439, upper bound: 0.0106080
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096439, upper bound: 0.0106080
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008706, 0.0008468
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024145, 0.0023034
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032078, 0.0033742
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023016, 0.0024268
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029664, 0.0030818
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022868, 0.0024118
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042533, 0.0039821
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0160019, 0.0152250
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035759, 0.0038042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104665, upper bound: 0.0096194
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104665, upper bound: 0.0096194
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008770, 0.0008436
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024386, 0.0022885
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031855, 0.0034104
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022849, 0.0024541
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029509, 0.0031069
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022701, 0.0024391
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0043217, 0.0039458
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0161716, 0.0151209
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035453, 0.0038590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105449, upper bound: 0.0096501
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105729, upper bound: 0.0096471
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009101, 0.0009060
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024644, 0.0024454
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034575, 0.0034859
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025088, 0.0025301
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030626, 0.0030823
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024955, 0.0025168
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046786, 0.0046323
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164820, 0.0163495
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040816, 0.0041205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009107, 0.0009052
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024672, 0.0024416
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034518, 0.0034902
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025045, 0.0025333
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030586, 0.0030852
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024912, 0.0025200
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046855, 0.0046230
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0165020, 0.0163227
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040737, 0.0041264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100422, upper bound: 0.0109149
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110586, upper bound: 0.0098862
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009266, 0.0009275
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025881, 0.0025921
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036801, 0.0036741
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026771, 0.0026726
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031968, 0.0031926
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026637, 0.0026591
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050040, 0.0050139
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0173576, 0.0173858
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043963, 0.0043880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102312, upper bound: 0.0101739
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009329, 0.0009244
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026128, 0.0025779
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036589, 0.0037125
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026612, 0.0027013
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031821, 0.0032176
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026477, 0.0026878
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050766, 0.0049793
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175369, 0.0172867
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043671, 0.0044420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102312, upper bound: 0.0101739
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009268, 0.0009299
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025891, 0.0026030
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036964, 0.0036757
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026893, 0.0026738
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032081, 0.0031937
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026758, 0.0026603
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050066, 0.0050403
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0173648, 0.0174616
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044185, 0.0043901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090355, upper bound: 0.0097588
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098002, upper bound: 0.0089611
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009331, 0.0009267
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026138, 0.0025880
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036740, 0.0037141
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026725, 0.0027024
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031926, 0.0032187
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026591, 0.0026889
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050791, 0.0050039
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0175441, 0.0173574
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043879, 0.0044441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0085472, upper bound: 0.0085164
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0085472, upper bound: 0.0085164
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009255, 0.0009310
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025828, 0.0026082
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0037042, 0.0036662
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026952, 0.0026666
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0032135, 0.0031871
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026817, 0.0026532
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0049911, 0.0050531
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0173206, 0.0174982
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044293, 0.0043771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0085288, upper bound: 0.0085424
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0085288, upper bound: 0.0085424
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009318, 0.0009280
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0026075, 0.0025940
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036830, 0.0037046
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026793, 0.0026953
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031988, 0.0032121
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026658, 0.0026818
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0050637, 0.0050185
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0174999, 0.0173991
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0044002, 0.0044311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089789, upper bound: 0.0097979
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097669, upper bound: 0.0090195
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008574, 0.0008896
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023486, 0.0024995
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035164, 0.0032904
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025426, 0.0023726
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031489, 0.0029921
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025283, 0.0023586
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042279, 0.0045961
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155954, 0.0166506
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040808, 0.0037707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0075912, upper bound: 0.0080971
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0075912, upper bound: 0.0080971
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008831, 0.0008618
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024691, 0.0023693
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033214, 0.0034709
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023959, 0.0025083
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030136, 0.0031172
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023819, 0.0024941
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045219, 0.0042783
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164378, 0.0157401
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038132, 0.0040182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100231, upper bound: 0.0091336
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100940, upper bound: 0.0091136
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009158, 0.0009186
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025318, 0.0025447
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036057, 0.0035865
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026198, 0.0026054
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031517, 0.0031384
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026065, 0.0025921
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0048752, 0.0049065
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0169527, 0.0170424
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043025, 0.0042762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091152, upper bound: 0.0099843
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100197, upper bound: 0.0090360
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009146, 0.0009199
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025262, 0.0025510
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0036152, 0.0035781
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0026269, 0.0025991
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031582, 0.0031325
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0026136, 0.0025857
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0048615, 0.0049219
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0169134, 0.0170865
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0043155, 0.0042647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102916, upper bound: 0.0103808
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103489, upper bound: 0.0103217
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009103, 0.0009041
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024661, 0.0024410
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034499, 0.0034884
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025026, 0.0025320
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030598, 0.0030840
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024893, 0.0025187
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046827, 0.0046150
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164939, 0.0163149
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040662, 0.0041240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009090, 0.0009057
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024602, 0.0024487
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034615, 0.0034796
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025113, 0.0025254
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030678, 0.0030779
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024980, 0.0025121
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046684, 0.0046339
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164528, 0.0163691
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040821, 0.0041120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008473, 0.0008660
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022959, 0.0023892
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033364, 0.0031968
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023983, 0.0022935
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030556, 0.0029587
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023834, 0.0022787
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039737, 0.0041917
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0151744, 0.0158253
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037523, 0.0035660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097193, upper bound: 0.0105632
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096644, upper bound: 0.0106081
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008744, 0.0008411
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024226, 0.0022727
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031619, 0.0033864
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022671, 0.0024361
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029345, 0.0030903
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022524, 0.0024211
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042827, 0.0039073
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0160598, 0.0150107
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035129, 0.0038262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107227, upper bound: 0.0096274
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106855, upper bound: 0.0096804
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0085424, upper bound: 0.0085288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0085424, upper bound: 0.0085288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0109548, upper bound: 0.0109088
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0110176, upper bound: 0.0108608
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0090195, upper bound: 0.0097669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0090195, upper bound: 0.0097669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0096804, upper bound: 0.0106855
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0096804, upper bound: 0.0106855
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097696, upper bound: 0.0107738
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0098005, upper bound: 0.0107200
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0106080, upper bound: 0.0096112
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0106248, upper bound: 0.0095926
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097979, upper bound: 0.0089789
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0095086, upper bound: 0.0096387
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097959, upper bound: 0.0109401
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0107814, upper bound: 0.0099526
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0109035, upper bound: 0.0110025
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0109022, upper bound: 0.0110042
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0095790, upper bound: 0.0104017
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0095790, upper bound: 0.0104017
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0096439, upper bound: 0.0106080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0096439, upper bound: 0.0106080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0104665, upper bound: 0.0096194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0104665, upper bound: 0.0096194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0105449, upper bound: 0.0096501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0105729, upper bound: 0.0096471
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0100422, upper bound: 0.0109149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0110586, upper bound: 0.0098862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0102312, upper bound: 0.0101739
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0102312, upper bound: 0.0101739
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0090355, upper bound: 0.0097588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0098002, upper bound: 0.0089611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0085472, upper bound: 0.0085164
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0085472, upper bound: 0.0085164
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0085288, upper bound: 0.0085424
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0085288, upper bound: 0.0085424
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0089789, upper bound: 0.0097979
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097669, upper bound: 0.0090195
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0075912, upper bound: 0.0080971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0075912, upper bound: 0.0080971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0100231, upper bound: 0.0091336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0100940, upper bound: 0.0091136
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0091152, upper bound: 0.0099843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0100197, upper bound: 0.0090360
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0102916, upper bound: 0.0103808
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0103489, upper bound: 0.0103217
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101801, upper bound: 0.0102100
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0101646, upper bound: 0.0102518
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0097193, upper bound: 0.0105632
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0096644, upper bound: 0.0106081
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0107227, upper bound: 0.0096274
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 8, lower bound: -0.0106855, upper bound: 0.0096804

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008965, 0.0008936
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024025, 0.0023888
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033691, 0.0033897
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024393, 0.0024547
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030034, 0.0030177
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024259, 0.0024413
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045224, 0.0044890
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0160397, 0.0159440
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0039516, 0.0039797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096063, upper bound: 0.0105005
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105163, upper bound: 0.0095318
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008975, 0.0008923
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024071, 0.0023828
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033601, 0.0033965
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024325, 0.0024599
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029972, 0.0030224
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024191, 0.0024464
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045336, 0.0044743
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0160718, 0.0159019
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0039392, 0.0039892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096295, upper bound: 0.0104436
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105838, upper bound: 0.0095237
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008613, 0.0008933
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023760, 0.0025253
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035541, 0.0033306
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025694, 0.0024012
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031721, 0.0030170
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025548, 0.0023870
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042702, 0.0046345
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157830, 0.0168268
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041239, 0.0038172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089301, upper bound: 0.0097145
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089694, upper bound: 0.0096563
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008860, 0.0008654
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024918, 0.0023947
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033585, 0.0035039
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024222, 0.0025316
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030364, 0.0031372
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024080, 0.0025171
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045526, 0.0043156
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0165921, 0.0159132
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038554, 0.0040550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008661, 0.0008878
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023980, 0.0024996
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035157, 0.0033625
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025404, 0.0024245
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031454, 0.0030403
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025260, 0.0024102
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0043233, 0.0045718
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0159348, 0.0166472
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040712, 0.0038592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089301, upper bound: 0.0097145
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089694, upper bound: 0.0096563
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008908, 0.0008621
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025137, 0.0023791
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033352, 0.0035358
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024047, 0.0025548
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030203, 0.0031605
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023905, 0.0025403
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046057, 0.0042778
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167439, 0.0158048
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038236, 0.0040969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008331, 0.0008633
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022340, 0.0023754
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033156, 0.0031039
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023828, 0.0022235
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030412, 0.0028943
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023678, 0.0022089
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038129, 0.0041579
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0147401, 0.0157285
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037239, 0.0034334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099381
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099375
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008396, 0.0008604
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022581, 0.0023618
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032952, 0.0031401
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023674, 0.0022508
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030270, 0.0029194
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023525, 0.0022362
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038813, 0.0041246
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149097, 0.0156332
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036959, 0.0034882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099381
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099375
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008264, 0.0008518
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021999, 0.0023192
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032272, 0.0030485
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023162, 0.0021818
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029847, 0.0028607
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023014, 0.0021673
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037282, 0.0040193
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144866, 0.0153207
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036056, 0.0033605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090319, upper bound: 0.0098736
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090319, upper bound: 0.0098736
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008272, 0.0008508
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022038, 0.0023144
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032200, 0.0030543
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023108, 0.0021862
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029797, 0.0028647
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022960, 0.0021716
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037376, 0.0040076
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0145134, 0.0152871
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035958, 0.0033684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096086, upper bound: 0.0104436
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095860, upper bound: 0.0104500
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008788, 0.0008526
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024472, 0.0023243
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032543, 0.0034384
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023449, 0.0024834
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029690, 0.0030967
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023310, 0.0024692
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0044706, 0.0041706
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162855, 0.0154259
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037191, 0.0039717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104956, upper bound: 0.0095630
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105661, upper bound: 0.0095457
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008799, 0.0008528
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024522, 0.0023253
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032558, 0.0034459
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023461, 0.0024890
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029700, 0.0031019
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023321, 0.0024748
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0044828, 0.0041731
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0163204, 0.0154331
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037212, 0.0039820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008855, 0.0008667
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024895, 0.0024008
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033677, 0.0035005
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024292, 0.0025290
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030428, 0.0031349
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024149, 0.0025145
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045470, 0.0043307
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0165761, 0.0159564
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038682, 0.0040503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096612, upper bound: 0.0089285
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097462, upper bound: 0.0089053
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008904, 0.0008640
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025114, 0.0023883
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033489, 0.0035324
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024150, 0.0025522
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030297, 0.0031581
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024008, 0.0025377
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046001, 0.0043000
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0167279, 0.0158686
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038423, 0.0040922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096612, upper bound: 0.0089285
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097462, upper bound: 0.0089053
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008962, 0.0009016
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024094, 0.0024346
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034360, 0.0033983
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024888, 0.0024604
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030531, 0.0030269
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024753, 0.0024469
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045313, 0.0045928
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0160813, 0.0162575
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040374, 0.0039856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008965, 0.0009002
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024111, 0.0024283
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034267, 0.0034008
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024818, 0.0024623
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030466, 0.0030286
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024682, 0.0024488
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045354, 0.0045775
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0160931, 0.0162137
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040245, 0.0039891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0082696, upper bound: 0.0089137
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0087927, upper bound: 0.0083804
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009029, 0.0008986
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024368, 0.0024206
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034150, 0.0034411
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024730, 0.0024934
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030385, 0.0030533
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024595, 0.0024799
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046062, 0.0045586
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162799, 0.0161595
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040086, 0.0040503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009033, 0.0008978
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024385, 0.0024172
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034099, 0.0034436
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024692, 0.0024953
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030350, 0.0030551
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024557, 0.0024818
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046104, 0.0045503
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162918, 0.0161357
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040016, 0.0040538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008312, 0.0008652
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022258, 0.0023868
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033284, 0.0030873
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023922, 0.0022110
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030548, 0.0028877
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023773, 0.0021964
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037915, 0.0041842
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0146677, 0.0157929
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037444, 0.0034137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0082696, upper bound: 0.0089137
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0082696, upper bound: 0.0089137
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008575, 0.0008379
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023492, 0.0022591
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031371, 0.0032721
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022484, 0.0023499
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029221, 0.0030158
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022337, 0.0023351
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040925, 0.0038725
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155303, 0.0148998
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034819, 0.0036672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104446, upper bound: 0.0096233
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104443, upper bound: 0.0096479
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008939, 0.0008979
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023902, 0.0024098
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034006, 0.0033712
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024629, 0.0024409
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030252, 0.0030049
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024495, 0.0024274
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0044924, 0.0045402
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0159537, 0.0160907
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0039947, 0.0039544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008923, 0.0008995
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023827, 0.0024173
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034118, 0.0033600
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024714, 0.0024324
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030330, 0.0029971
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024579, 0.0024190
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0044740, 0.0045585
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0159011, 0.0161433
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040101, 0.0039390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008404, 0.0008708
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022712, 0.0024136
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033729, 0.0031596
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024258, 0.0022654
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030809, 0.0029329
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024108, 0.0022507
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039036, 0.0042512
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0150000, 0.0159959
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038025, 0.0035098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008394, 0.0008742
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022667, 0.0024316
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033999, 0.0031529
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024461, 0.0022604
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030996, 0.0029283
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024310, 0.0022457
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038927, 0.0042951
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149688, 0.0161217
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038394, 0.0035006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008405, 0.0008588
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022623, 0.0023542
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032839, 0.0031465
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023589, 0.0022556
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030192, 0.0029238
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023440, 0.0022410
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038917, 0.0041062
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149394, 0.0155805
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036804, 0.0034969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008391, 0.0008594
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022560, 0.0023569
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032880, 0.0031369
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023619, 0.0022484
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030220, 0.0029172
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023470, 0.0022338
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038761, 0.0041128
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148948, 0.0155994
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036859, 0.0034838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008680, 0.0008434
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024005, 0.0022854
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031808, 0.0033532
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022814, 0.0024110
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029477, 0.0030672
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022666, 0.0023960
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042190, 0.0039382
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0159037, 0.0150991
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035389, 0.0037754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008671, 0.0008468
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023965, 0.0023034
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032078, 0.0033472
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023016, 0.0024065
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029664, 0.0030631
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022868, 0.0023915
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042094, 0.0039821
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0158761, 0.0152250
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035759, 0.0037672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008632, 0.0008309
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023774, 0.0022293
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030927, 0.0033144
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022144, 0.0023817
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028907, 0.0030451
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021999, 0.0023668
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041614, 0.0037911
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157276, 0.0146937
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034118, 0.0037252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008647, 0.0008307
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023847, 0.0022285
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030917, 0.0033253
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022136, 0.0023899
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028899, 0.0030527
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021991, 0.0023750
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041791, 0.0037893
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157785, 0.0146887
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034103, 0.0037402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009033, 0.0009018
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024366, 0.0024295
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034327, 0.0034434
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024896, 0.0024977
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030478, 0.0030553
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024764, 0.0024844
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046044, 0.0045870
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162846, 0.0162347
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040426, 0.0040572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089559, upper bound: 0.0097065
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096631, upper bound: 0.0089102
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009101, 0.0008993
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024644, 0.0024177
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034150, 0.0034859
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024764, 0.0025301
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030356, 0.0030823
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024631, 0.0025168
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046786, 0.0045582
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164820, 0.0161521
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040183, 0.0041205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089559, upper bound: 0.0097065
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096631, upper bound: 0.0089102
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008404, 0.0008612
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022608, 0.0023586
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032906, 0.0031442
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023640, 0.0022539
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030238, 0.0029223
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023492, 0.0022393
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038881, 0.0041266
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149291, 0.0156124
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036947, 0.0034939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096278, upper bound: 0.0104874
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096479, upper bound: 0.0104443
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008655, 0.0008349
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023784, 0.0022352
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031058, 0.0033203
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022251, 0.0023863
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028957, 0.0030444
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022105, 0.0023714
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041749, 0.0038255
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157510, 0.0147498
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034412, 0.0037354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105711, upper bound: 0.0095227
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105833, upper bound: 0.0095023
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009019, 0.0009029
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024300, 0.0024348
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034406, 0.0034334
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024956, 0.0024902
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030533, 0.0030483
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024823, 0.0024769
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045882, 0.0045999
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162380, 0.0162717
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040535, 0.0040436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009025, 0.0009028
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024329, 0.0024340
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034394, 0.0034378
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024947, 0.0024935
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030525, 0.0030514
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024815, 0.0024802
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045953, 0.0045980
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162583, 0.0162662
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040518, 0.0040495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009087, 0.0009004
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024577, 0.0024228
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034227, 0.0034759
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024821, 0.0025226
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030409, 0.0030753
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024689, 0.0025093
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046623, 0.0045707
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164354, 0.0161879
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040288, 0.0041068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089053, upper bound: 0.0097462
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096563, upper bound: 0.0089694
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009093, 0.0008997
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024606, 0.0024198
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034182, 0.0034803
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024788, 0.0025259
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030378, 0.0030784
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024655, 0.0025126
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046694, 0.0045634
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164557, 0.0161670
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040227, 0.0041128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097145, upper bound: 0.0089301
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008618, 0.0008934
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023785, 0.0025259
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035550, 0.0033343
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025700, 0.0024040
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031727, 0.0030196
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025555, 0.0023898
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042762, 0.0046359
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0158004, 0.0168308
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0041251, 0.0038223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008857, 0.0008649
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024902, 0.0023924
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033550, 0.0035016
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024196, 0.0025298
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030340, 0.0031357
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024054, 0.0025154
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045489, 0.0043100
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0165815, 0.0158971
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038507, 0.0040519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079610, upper bound: 0.0074288
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079610, upper bound: 0.0074288
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008653, 0.0008891
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023942, 0.0025056
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0035245, 0.0033567
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025471, 0.0024201
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031516, 0.0030363
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0025326, 0.0024059
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0043139, 0.0045862
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0159079, 0.0166886
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040833, 0.0038512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089053, upper bound: 0.0097462
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008894, 0.0008630
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0025068, 0.0023834
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033416, 0.0035254
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024095, 0.0025469
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030247, 0.0031533
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023953, 0.0025325
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0045887, 0.0042882
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0166953, 0.0158346
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0038323, 0.0040827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079412, upper bound: 0.0074594
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079412, upper bound: 0.0074594
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008524, 0.0008319
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023264, 0.0022306
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030947, 0.0032382
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022160, 0.0023238
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028921, 0.0029916
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022014, 0.0023091
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040281, 0.0037943
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0153729, 0.0147030
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034145, 0.0036114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079690, upper bound: 0.0075407
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079690, upper bound: 0.0075407
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008550, 0.0008311
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023386, 0.0022266
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030888, 0.0032564
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022115, 0.0023375
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028879, 0.0030042
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021969, 0.0023227
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040578, 0.0037846
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0154579, 0.0146752
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034064, 0.0036364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0080488, upper bound: 0.0075213
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0080488, upper bound: 0.0075213
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008514, 0.0008806
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023190, 0.0024560
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034512, 0.0032461
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024935, 0.0023393
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0031036, 0.0029614
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024793, 0.0023254
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041557, 0.0044899
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0153886, 0.0163461
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0039913, 0.0037099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008780, 0.0008541
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024435, 0.0023318
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032653, 0.0034325
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023537, 0.0024795
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029747, 0.0030907
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023398, 0.0024653
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0044594, 0.0041870
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0162589, 0.0154783
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037362, 0.0039657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099023, upper bound: 0.0089835
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099693, upper bound: 0.0089707
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008891, 0.0008945
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023732, 0.0023983
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033817, 0.0033441
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024480, 0.0024197
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030154, 0.0029893
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024345, 0.0024062
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0044430, 0.0045043
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0158282, 0.0160040
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0039629, 0.0039112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008907, 0.0008944
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023805, 0.0023980
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033812, 0.0033550
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024475, 0.0024279
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030150, 0.0029969
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024341, 0.0024145
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0044608, 0.0045034
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0158794, 0.0160013
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0039621, 0.0039263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009080, 0.0009006
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024544, 0.0024241
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034246, 0.0034710
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024835, 0.0025189
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030422, 0.0030719
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024703, 0.0025056
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046542, 0.0045738
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0164123, 0.0161967
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040314, 0.0041000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089559, upper bound: 0.0097065
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096631, upper bound: 0.0089102
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009068, 0.0009041
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024492, 0.0024410
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034499, 0.0034631
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025026, 0.0025130
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030598, 0.0030665
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024893, 0.0024997
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046415, 0.0046150
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0163757, 0.0163149
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040662, 0.0040893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084320, upper bound: 0.0084662
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084320, upper bound: 0.0084662
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009067, 0.0009023
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024485, 0.0024318
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034362, 0.0034621
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0024923, 0.0025122
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030503, 0.0030657
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024790, 0.0024990
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046398, 0.0045927
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0163708, 0.0162509
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040473, 0.0040878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0009056, 0.0009057
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0024433, 0.0024487
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0034615, 0.0034543
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0025113, 0.0025064
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030678, 0.0030604
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0024980, 0.0024931
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0046271, 0.0046339
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0163346, 0.0163691
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0040821, 0.0040772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089053, upper bound: 0.0097462
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096563, upper bound: 0.0089694
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008408, 0.0008586
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022639, 0.0023531
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032823, 0.0031488
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023577, 0.0022574
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030180, 0.0029255
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023428, 0.0022427
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038956, 0.0041035
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149505, 0.0155729
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036781, 0.0035002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089851, upper bound: 0.0096565
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089851, upper bound: 0.0096565
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008400, 0.0008597
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022598, 0.0023585
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032902, 0.0031427
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023637, 0.0022528
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030236, 0.0029212
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023487, 0.0022382
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038856, 0.0041165
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149220, 0.0156100
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036891, 0.0034918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008676, 0.0008337
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023894, 0.0022366
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031078, 0.0033368
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022265, 0.0023987
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028970, 0.0030558
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022118, 0.0023838
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0042018, 0.0038192
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0158279, 0.0147582
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034387, 0.0037580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097484, upper bound: 0.0088883
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097484, upper bound: 0.0088883
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008670, 0.0008350
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023865, 0.0022427
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031169, 0.0033324
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022333, 0.0023954
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029034, 0.0030528
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022187, 0.0023805
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041946, 0.0038341
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0158073, 0.0148009
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034513, 0.0037520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099375, upper bound: 0.0090118
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099381, upper bound: 0.0090118
time: 0.65 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096063, upper bound: 0.0105005
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0105163, upper bound: 0.0095318
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096295, upper bound: 0.0104436
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0105838, upper bound: 0.0095237
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089301, upper bound: 0.0097145
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089694, upper bound: 0.0096563
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089301, upper bound: 0.0097145
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089694, upper bound: 0.0096563
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099381
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099375
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099381
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0090118, upper bound: 0.0099375
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0090319, upper bound: 0.0098736
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0090319, upper bound: 0.0098736
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096086, upper bound: 0.0104436
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0095860, upper bound: 0.0104500
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0104956, upper bound: 0.0095630
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0105661, upper bound: 0.0095457
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079607, upper bound: 0.0074390
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096612, upper bound: 0.0089285
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097462, upper bound: 0.0089053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096612, upper bound: 0.0089285
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097462, upper bound: 0.0089053
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0082696, upper bound: 0.0089137
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0087927, upper bound: 0.0083804
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0082696, upper bound: 0.0089137
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0082696, upper bound: 0.0089137
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0104446, upper bound: 0.0096233
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0104443, upper bound: 0.0096479
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084010, upper bound: 0.0084978
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089102, upper bound: 0.0096631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097065, upper bound: 0.0089559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0088984, upper bound: 0.0083423
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089559, upper bound: 0.0097065
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096631, upper bound: 0.0089102
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089559, upper bound: 0.0097065
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096631, upper bound: 0.0089102
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096278, upper bound: 0.0104874
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096479, upper bound: 0.0104443
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0105711, upper bound: 0.0095227
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0105833, upper bound: 0.0095023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089053, upper bound: 0.0097462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096563, upper bound: 0.0089694
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097145, upper bound: 0.0089301
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079610, upper bound: 0.0074288
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079610, upper bound: 0.0074288
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089053, upper bound: 0.0097462
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079412, upper bound: 0.0074594
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079412, upper bound: 0.0074594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079690, upper bound: 0.0075407
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0079690, upper bound: 0.0075407
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0080488, upper bound: 0.0075213
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0080488, upper bound: 0.0075213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0074594, upper bound: 0.0079321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0099023, upper bound: 0.0089835
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0099693, upper bound: 0.0089707
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084785, upper bound: 0.0084161
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089559, upper bound: 0.0097065
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096631, upper bound: 0.0089102
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084320, upper bound: 0.0084662
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084320, upper bound: 0.0084662
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0084207, upper bound: 0.0084928
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089053, upper bound: 0.0097462
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0096563, upper bound: 0.0089694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089851, upper bound: 0.0096565
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089851, upper bound: 0.0096565
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0089285, upper bound: 0.0096612
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097484, upper bound: 0.0088883
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0097484, upper bound: 0.0088883
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0099375, upper bound: 0.0090118
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 8, lower bound: -0.0099381, upper bound: 0.0090118

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008256, 0.0008497
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021955, 0.0023084
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032110, 0.0030419
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023040, 0.0021768
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029735, 0.0028561
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022892, 0.0021623
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037174, 0.0039930
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144556, 0.0152452
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035834, 0.0033514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008505, 0.0008227
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023120, 0.0021818
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030214, 0.0032163
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021614, 0.0023080
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028419, 0.0029771
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021469, 0.0022932
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040016, 0.0036840
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0152698, 0.0143598
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033232, 0.0035907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008266, 0.0008471
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022001, 0.0022963
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031928, 0.0030488
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022903, 0.0021820
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029608, 0.0028609
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022756, 0.0021675
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037286, 0.0039633
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144877, 0.0151601
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035584, 0.0033608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008535, 0.0008214
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023261, 0.0021758
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030124, 0.0032374
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021546, 0.0023239
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028357, 0.0029918
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021401, 0.0023091
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040360, 0.0036693
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0153684, 0.0143178
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033109, 0.0036196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008316, 0.0008637
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022258, 0.0023769
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033179, 0.0030916
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023845, 0.0022143
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030428, 0.0028858
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023695, 0.0021997
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037929, 0.0041616
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0146828, 0.0157393
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037271, 0.0034165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008323, 0.0008634
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022291, 0.0023759
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033164, 0.0030966
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023833, 0.0022180
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030417, 0.0028892
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023684, 0.0022034
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038009, 0.0041592
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0147057, 0.0157322
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037250, 0.0034233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008380, 0.0008605
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022499, 0.0023624
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032962, 0.0031278
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023681, 0.0022416
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030277, 0.0029109
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023532, 0.0022270
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038613, 0.0041262
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148524, 0.0156377
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036972, 0.0034713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008387, 0.0008579
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022532, 0.0023502
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032780, 0.0031327
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023544, 0.0022453
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030150, 0.0029143
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023395, 0.0022307
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038693, 0.0040965
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148754, 0.0155526
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036722, 0.0034781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008202, 0.0008524
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021741, 0.0023249
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032360, 0.0030101
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023222, 0.0021523
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029901, 0.0028334
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023074, 0.0021378
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0036564, 0.0040245
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0143078, 0.0153625
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036084, 0.0032984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008207, 0.0008504
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021763, 0.0023155
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032218, 0.0030135
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023115, 0.0021549
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029802, 0.0028357
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022968, 0.0021404
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0036619, 0.0040014
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0143237, 0.0152963
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035889, 0.0033031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008257, 0.0008484
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021970, 0.0023061
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032078, 0.0030441
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023009, 0.0021785
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029705, 0.0028577
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022862, 0.0021639
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037210, 0.0039785
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144658, 0.0152307
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035696, 0.0033544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008262, 0.0008475
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021992, 0.0023018
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032014, 0.0030475
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022962, 0.0021810
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029661, 0.0028600
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022814, 0.0021665
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037265, 0.0039681
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144816, 0.0152010
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035609, 0.0033590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008208, 0.0008514
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021771, 0.0023201
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032287, 0.0030146
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023167, 0.0021557
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029850, 0.0028365
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023019, 0.0021412
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0036637, 0.0040127
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0143287, 0.0153286
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035984, 0.0033045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008264, 0.0008463
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021999, 0.0022964
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031932, 0.0030485
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022900, 0.0021818
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029604, 0.0028607
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022753, 0.0021673
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037282, 0.0039548
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144866, 0.0151627
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035497, 0.0033605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008248, 0.0008474
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021917, 0.0022975
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031947, 0.0030361
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022917, 0.0021725
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029621, 0.0028522
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022770, 0.0021580
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037080, 0.0039664
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144287, 0.0151689
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035610, 0.0033435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008238, 0.0008508
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021869, 0.0023144
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032200, 0.0030290
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023108, 0.0021671
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029797, 0.0028472
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022960, 0.0021526
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0036963, 0.0040076
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0143952, 0.0152871
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035958, 0.0033336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008494, 0.0008237
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023068, 0.0021867
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030288, 0.0032086
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021670, 0.0023022
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028470, 0.0029718
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021524, 0.0022874
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039891, 0.0036960
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0152339, 0.0143943
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033334, 0.0035801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008512, 0.0008231
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023154, 0.0021839
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030245, 0.0032214
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021637, 0.0023118
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028441, 0.0029806
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021492, 0.0022970
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040099, 0.0036891
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0152936, 0.0143743
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033275, 0.0035977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008558, 0.0008367
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023393, 0.0022509
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031292, 0.0032615
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022425, 0.0023421
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029118, 0.0030036
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022278, 0.0023272
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040697, 0.0038540
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0154759, 0.0148579
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034680, 0.0036496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008577, 0.0008368
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023483, 0.0022515
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031300, 0.0032751
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022432, 0.0023523
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029124, 0.0030131
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022285, 0.0023374
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040918, 0.0038554
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155392, 0.0148619
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034692, 0.0036683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008623, 0.0008346
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023633, 0.0022410
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031143, 0.0032977
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022314, 0.0023693
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029016, 0.0030287
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022167, 0.0023545
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041381, 0.0038299
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156455, 0.0147888
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034477, 0.0037044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008642, 0.0008342
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023724, 0.0022389
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031112, 0.0033113
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022290, 0.0023795
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028994, 0.0030382
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022143, 0.0023647
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041603, 0.0038247
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157089, 0.0147740
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034434, 0.0037231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008269, 0.0008583
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022087, 0.0023559
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032824, 0.0030619
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023571, 0.0021913
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030223, 0.0028693
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023422, 0.0021767
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037408, 0.0041002
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0145497, 0.0155793
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036721, 0.0033695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008257, 0.0008641
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022030, 0.0023848
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033257, 0.0030533
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023896, 0.0021848
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030523, 0.0028634
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023747, 0.0021703
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037269, 0.0041707
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0145098, 0.0157813
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037315, 0.0033578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008312, 0.0008596
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022258, 0.0023639
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032944, 0.0030873
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023661, 0.0022110
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030305, 0.0028877
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023512, 0.0021964
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037915, 0.0041196
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0146677, 0.0156350
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036885, 0.0034137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008477, 0.0008277
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022992, 0.0022062
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030579, 0.0031971
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021889, 0.0022936
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028672, 0.0029638
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021743, 0.0022788
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039704, 0.0037435
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0151804, 0.0145303
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033733, 0.0035644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008474, 0.0008287
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022975, 0.0022111
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030652, 0.0031946
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021944, 0.0022917
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028723, 0.0029620
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021798, 0.0022769
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039662, 0.0037554
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0151684, 0.0145644
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033833, 0.0035609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008318, 0.0008606
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022271, 0.0023619
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032954, 0.0030935
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023675, 0.0022157
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030271, 0.0028871
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023526, 0.0022011
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037960, 0.0041249
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0146917, 0.0156340
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036961, 0.0034192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008302, 0.0008598
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022195, 0.0023582
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032898, 0.0030821
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023633, 0.0022071
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030233, 0.0028792
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023484, 0.0021925
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037773, 0.0041158
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0146382, 0.0156081
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036885, 0.0034034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008306, 0.0008640
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022214, 0.0023788
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033207, 0.0030850
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023866, 0.0022093
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030447, 0.0028812
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023716, 0.0021947
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037820, 0.0041661
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0146516, 0.0157522
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037308, 0.0034074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008292, 0.0008633
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022150, 0.0023751
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033152, 0.0030754
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023824, 0.0022021
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030408, 0.0028746
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023674, 0.0021875
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037664, 0.0041571
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0146070, 0.0157263
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037232, 0.0033943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008383, 0.0008554
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022512, 0.0023373
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032586, 0.0031297
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023399, 0.0022430
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030016, 0.0029122
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023250, 0.0022284
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038644, 0.0040650
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148614, 0.0154623
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036456, 0.0034740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008371, 0.0008588
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022454, 0.0023542
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032839, 0.0031211
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023589, 0.0022366
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030192, 0.0029063
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023440, 0.0022220
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038504, 0.0041062
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148212, 0.0155805
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036804, 0.0034622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008367, 0.0008560
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022435, 0.0023400
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032627, 0.0031183
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023429, 0.0022344
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030044, 0.0029043
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023280, 0.0022198
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038458, 0.0040716
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148078, 0.0154812
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036512, 0.0034582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008357, 0.0008594
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022390, 0.0023569
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032880, 0.0031116
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023619, 0.0022294
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030220, 0.0028996
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023470, 0.0022148
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038349, 0.0041128
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0147766, 0.0155994
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036859, 0.0034491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008582, 0.0008332
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023506, 0.0022336
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031033, 0.0032784
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022231, 0.0023548
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028939, 0.0030154
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022084, 0.0023399
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040973, 0.0038119
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155549, 0.0147372
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034325, 0.0036728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008578, 0.0008348
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023487, 0.0022409
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031143, 0.0032757
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022313, 0.0023527
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029015, 0.0030134
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022167, 0.0023378
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040927, 0.0038297
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155419, 0.0147884
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034476, 0.0036690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008571, 0.0008366
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023453, 0.0022505
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031286, 0.0032705
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022421, 0.0023488
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029115, 0.0030099
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022274, 0.0023340
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040844, 0.0038531
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155180, 0.0148554
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034673, 0.0036620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008570, 0.0008382
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023447, 0.0022578
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031396, 0.0032697
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022504, 0.0023482
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029191, 0.0030093
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022357, 0.0023333
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040831, 0.0038710
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155142, 0.0149066
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034823, 0.0036609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008606, 0.0008274
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023634, 0.0022113
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030658, 0.0032933
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021942, 0.0023659
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028720, 0.0030305
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021796, 0.0023510
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041271, 0.0037471
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156294, 0.0145678
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033748, 0.0036963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008597, 0.0008309
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023594, 0.0022293
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030927, 0.0032874
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022144, 0.0023615
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028907, 0.0030264
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021999, 0.0023466
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041175, 0.0037911
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156017, 0.0146937
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034118, 0.0036882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008624, 0.0008273
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023721, 0.0022105
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030647, 0.0033064
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021934, 0.0023757
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028712, 0.0030396
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021788, 0.0023608
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041484, 0.0037454
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156903, 0.0145628
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033733, 0.0037142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008613, 0.0008307
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023667, 0.0022285
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030917, 0.0032983
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022136, 0.0023696
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028899, 0.0030340
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021991, 0.0023547
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041352, 0.0037893
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156526, 0.0146887
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034103, 0.0037032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008333, 0.0008607
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022339, 0.0023621
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032957, 0.0031038
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023678, 0.0022234
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030274, 0.0028942
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023529, 0.0022088
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038127, 0.0041255
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0147394, 0.0156357
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036966, 0.0034332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008572, 0.0008318
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023459, 0.0022268
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030931, 0.0032714
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022154, 0.0023495
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028868, 0.0030105
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022008, 0.0023346
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040858, 0.0037952
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0155219, 0.0146895
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034185, 0.0036632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008398, 0.0008570
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022580, 0.0023447
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032697, 0.0031400
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023482, 0.0022507
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030093, 0.0029193
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023333, 0.0022361
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038811, 0.0040831
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0149091, 0.0155142
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036609, 0.0034880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008637, 0.0008292
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023699, 0.0022150
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030754, 0.0033076
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022021, 0.0023768
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028746, 0.0030356
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021875, 0.0023619
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041542, 0.0037664
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156916, 0.0146070
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033943, 0.0037180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008265, 0.0008492
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021997, 0.0023060
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032074, 0.0030482
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023013, 0.0021816
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029710, 0.0028605
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022865, 0.0021670
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037277, 0.0039872
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144851, 0.0152284
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035785, 0.0033600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008265, 0.0008474
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0021998, 0.0022975
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031946, 0.0030483
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022917, 0.0021816
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029620, 0.0028606
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022769, 0.0021671
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0037278, 0.0039662
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0144854, 0.0151684
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0035609, 0.0033601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008516, 0.0008214
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023173, 0.0021757
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030122, 0.0032243
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021545, 0.0023140
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028355, 0.0029826
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021400, 0.0022992
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040146, 0.0036690
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0153070, 0.0143169
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033106, 0.0036016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008527, 0.0008210
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023223, 0.0021741
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030098, 0.0032318
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021527, 0.0023197
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028339, 0.0029879
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021382, 0.0023049
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040269, 0.0036652
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0153423, 0.0143059
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033074, 0.0036120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008383, 0.0008577
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022513, 0.0023483
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032751, 0.0031300
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023523, 0.0022432
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030131, 0.0029124
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023374, 0.0022286
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038648, 0.0040918
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148625, 0.0155392
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036683, 0.0034743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008626, 0.0008303
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023651, 0.0022201
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030831, 0.0033004
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022079, 0.0023713
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028799, 0.0030306
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021932, 0.0023565
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041425, 0.0037789
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156579, 0.0146427
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034048, 0.0037081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008390, 0.0008558
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022542, 0.0023393
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032615, 0.0031343
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023421, 0.0022465
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030036, 0.0029154
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023272, 0.0022319
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038719, 0.0040697
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148828, 0.0154759
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036496, 0.0034803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008643, 0.0008297
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023730, 0.0022171
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030786, 0.0033121
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022045, 0.0023802
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028768, 0.0030387
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021899, 0.0023653
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041616, 0.0037716
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157129, 0.0146219
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033986, 0.0037242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008372, 0.0008612
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022460, 0.0023652
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033004, 0.0031221
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023713, 0.0022373
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030306, 0.0029069
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023564, 0.0022226
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038519, 0.0041331
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148255, 0.0156574
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037030, 0.0034634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008377, 0.0008592
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022481, 0.0023562
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032868, 0.0031252
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023611, 0.0022396
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030212, 0.0029091
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023462, 0.0022250
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038571, 0.0041109
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148402, 0.0155940
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036844, 0.0034678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008473, 0.0008234
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023009, 0.0021892
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030327, 0.0032000
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021693, 0.0022951
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028490, 0.0029651
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021548, 0.0022804
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039658, 0.0036932
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0151944, 0.0144132
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033294, 0.0035590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008495, 0.0008234
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023110, 0.0021892
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030328, 0.0032152
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021694, 0.0023065
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028491, 0.0029756
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021549, 0.0022918
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0039906, 0.0036934
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0152655, 0.0144137
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033295, 0.0035799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008376, 0.0008571
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022480, 0.0023453
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032705, 0.0031250
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023488, 0.0022395
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030099, 0.0029090
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023340, 0.0022249
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038568, 0.0040844
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148394, 0.0155180
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036620, 0.0034675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008628, 0.0008306
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023657, 0.0022214
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030850, 0.0033013
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022093, 0.0023720
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028812, 0.0030312
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021947, 0.0023572
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041440, 0.0037820
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156622, 0.0146516
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034074, 0.0037093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008353, 0.0008615
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022369, 0.0023670
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0033031, 0.0031084
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023733, 0.0022270
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030325, 0.0028974
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023584, 0.0022124
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038297, 0.0041374
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0147617, 0.0156699
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0037066, 0.0034447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008610, 0.0008357
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023574, 0.0022460
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031219, 0.0032888
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022370, 0.0023627
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0029068, 0.0030226
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022224, 0.0023478
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041237, 0.0038421
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0156041, 0.0148239
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034580, 0.0036923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008386, 0.0008551
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022525, 0.0023362
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032570, 0.0031317
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023386, 0.0022445
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030005, 0.0029136
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023238, 0.0022299
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038676, 0.0040623
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148704, 0.0154547
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036434, 0.0034766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008374, 0.0008586
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022470, 0.0023531
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032823, 0.0031235
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023577, 0.0022384
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030180, 0.0029079
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023428, 0.0022237
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038543, 0.0041035
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148323, 0.0155729
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036781, 0.0034654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008374, 0.0008563
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022469, 0.0023415
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032649, 0.0031233
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023446, 0.0022382
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030060, 0.0029078
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023297, 0.0022236
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038539, 0.0040753
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148313, 0.0154918
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036543, 0.0034651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008365, 0.0008597
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0022429, 0.0023585
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0032902, 0.0031174
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0023637, 0.0022338
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0030236, 0.0029037
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0023487, 0.0022192
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0038444, 0.0041165
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0148038, 0.0156100
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0036891, 0.0034570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008650, 0.0008303
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023763, 0.0022197
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030825, 0.0033171
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022074, 0.0023839
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028795, 0.0030422
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021928, 0.0023690
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041697, 0.0037780
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157360, 0.0146401
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034040, 0.0037310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008642, 0.0008337
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023725, 0.0022366
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0031078, 0.0033115
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0022265, 0.0023797
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028970, 0.0030383
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0022118, 0.0023648
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0041605, 0.0038192
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0157097, 0.0147582
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0034387, 0.0037233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008531, 0.0008224
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023253, 0.0021846
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030259, 0.0032363
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021642, 0.0023231
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028443, 0.0029910
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021497, 0.0023082
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040342, 0.0036821
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0153634, 0.0143815
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033201, 0.0036182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005210, 0.0007656, -0.0005210, 0.0007656, -0.0008544, 0.0008221
1: -0.0011187, 0.0024950, -0.0011187, 0.0024950, -0.0023313, 0.0021828
2: 0.0126035, 0.0180154, 0.0126035, 0.0180154, -0.0030231, 0.0032452
3: -0.0011496, 0.0029199, -0.0011496, 0.0029199, -0.0021621, 0.0023297
4: -0.0054400, -0.0016863, -0.0054400, -0.0016863, -0.0028424, 0.0029972
5: 0.0067906, 0.0108528, 0.0067906, 0.0108528, -0.0021476, 0.0023149
6: 0.0080945, 0.0103703, 0.0080945, 0.0103703, -0.0022758, 0.0022758
7: -0.0219596, -0.0131412, -0.0219596, -0.0131412, -0.0040487, 0.0036777
8: 0.9608741, 0.9861398, 0.9608741, 0.9861398, -0.0154048, 0.0143687
9: 0.0017267, 0.0091523, 0.0017267, 0.0091523, -0.0033163, 0.0036303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
time: 0.63 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073917, upper bound: 0.0078914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078215
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078244, upper bound: 0.0073886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079113, upper bound: 0.0073739
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073632, upper bound: 0.0079117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078077, upper bound: 0.0074092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073783, upper bound: 0.0078326
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078825, upper bound: 0.0073941
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073941, upper bound: 0.0078825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078326, upper bound: 0.0073783
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073739, upper bound: 0.0079113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078215, upper bound: 0.0074092
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0074092, upper bound: 0.0078077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0073886, upper bound: 0.0078244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0079117, upper bound: 0.0073632
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 8, lower bound: -0.0078914, upper bound: 0.0073917

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.73 + 499.07 = 501.80 seconds
