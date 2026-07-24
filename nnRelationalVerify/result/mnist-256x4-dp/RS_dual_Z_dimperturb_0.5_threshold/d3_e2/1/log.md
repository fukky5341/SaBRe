## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03428451


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156367, 0.0156367)
1: (-0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261)
2: (0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0397224, 0.0397224)
3: (-0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0191173, 0.0191173)
4: (-0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136)
5: (0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782)
6: (0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269)
7: (-0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0309872, 0.0309872)
8: (0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0734371)
9: (-0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 2.01 = 3.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0394965, upper bound: 0.0394965

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0384754, upper bound: 0.0387466
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0387466, upper bound: 0.0384754
time: 1.05 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 8, lower bound: -0.0384754, upper bound: 0.0387466
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 8, lower bound: -0.0387466, upper bound: 0.0384754

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155937, 0.0156120
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0395473, 0.0394195
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0189985, 0.0189023
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0305197, 0.0307280
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0380794, upper bound: 0.0384939
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0382416, upper bound: 0.0384598
time: 1.16 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156367, 0.0155937
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0394195, 0.0397224
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0189023, 0.0191173
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0309872, 0.0305197
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0384598, upper bound: 0.0382416
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0384939, upper bound: 0.0380794
time: 1.24 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 8, lower bound: -0.0380794, upper bound: 0.0384939
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 8, lower bound: -0.0382416, upper bound: 0.0384598
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 8, lower bound: -0.0384598, upper bound: 0.0382416
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 8, lower bound: -0.0384939, upper bound: 0.0380794

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155627, 0.0155788
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0392468, 0.0391340
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0187250, 0.0186402
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0299290, 0.0301129
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366441, upper bound: 0.0370117
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366713, upper bound: 0.0370021
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155618, 0.0155809
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0392618, 0.0391273
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0187363, 0.0186352
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0299182, 0.0301374
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367489, upper bound: 0.0369787
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367856, upper bound: 0.0369719
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156067, 0.0155618
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0391273, 0.0394382
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0186352, 0.0188626
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0304136, 0.0299182
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369719, upper bound: 0.0367856
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369787, upper bound: 0.0367489
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156058, 0.0155627
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0391340, 0.0394316
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0186402, 0.0188577
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0304028, 0.0299290
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370021, upper bound: 0.0366713
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370117, upper bound: 0.0366441
time: 1.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0366441, upper bound: 0.0370117
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0366713, upper bound: 0.0370021
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0367489, upper bound: 0.0369787
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0367856, upper bound: 0.0369719
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0369719, upper bound: 0.0367856
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0369787, upper bound: 0.0367489
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0370021, upper bound: 0.0366713
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0370117, upper bound: 0.0366441

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155208, 0.0155522
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0390549, 0.0388349
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0185735, 0.0184081
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0294423, 0.0298008
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0727448, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365682, upper bound: 0.0369583
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365678, upper bound: 0.0369584
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155627, 0.0155369
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389478, 0.0391340
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184929, 0.0186402
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0299290, 0.0296262
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0732718
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366023, upper bound: 0.0369443
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366022, upper bound: 0.0369442
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155199, 0.0155519
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0390523, 0.0388283
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0185715, 0.0184031
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0294315, 0.0297965
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0727139, 0.0734371
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366774, upper bound: 0.0369237
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366774, upper bound: 0.0369234
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155618, 0.0155391
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389627, 0.0391273
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0185042, 0.0186352
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0299182, 0.0296506
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0733417
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367210, upper bound: 0.0369124
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367220, upper bound: 0.0369118
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155650, 0.0155340
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389272, 0.0391386
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184775, 0.0186322
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0299220, 0.0295928
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0731760
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369118, upper bound: 0.0367220
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369124, upper bound: 0.0367210
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156067, 0.0155199
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388283, 0.0394382
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184031, 0.0188626
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0304136, 0.0294315
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0727139
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369234, upper bound: 0.0366774
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369237, upper bound: 0.0366774
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155640, 0.0155307
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389043, 0.0391320
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184603, 0.0186272
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0299112, 0.0295554
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0730689
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369442, upper bound: 0.0366022
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369443, upper bound: 0.0366023
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156058, 0.0155208
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388349, 0.0394316
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184081, 0.0188577
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0304028, 0.0294423
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0727448
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369584, upper bound: 0.0365678
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369583, upper bound: 0.0365682
time: 1.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0365682, upper bound: 0.0369583
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0365678, upper bound: 0.0369584
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0366023, upper bound: 0.0369443
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0366022, upper bound: 0.0369442
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0366774, upper bound: 0.0369237
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0366774, upper bound: 0.0369234
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0367210, upper bound: 0.0369124
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0367220, upper bound: 0.0369118
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369118, upper bound: 0.0367220
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369124, upper bound: 0.0367210
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369234, upper bound: 0.0366774
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369237, upper bound: 0.0366774
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369442, upper bound: 0.0366022
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369443, upper bound: 0.0366023
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369584, upper bound: 0.0365678
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 8, lower bound: -0.0369583, upper bound: 0.0365682

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155150, 0.0155456
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105232, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389922, 0.0387774
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0185248, 0.0183632
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0293235, 0.0296736
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0723908, 0.0733939
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361097, upper bound: 0.0365292
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361185, upper bound: 0.0365294
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155142, 0.0155455
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105198, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389918, 0.0387722
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0185245, 0.0183594
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0293151, 0.0296729
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0723668, 0.0733918
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361092, upper bound: 0.0365293
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361181, upper bound: 0.0365294
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155568, 0.0155303
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388851, 0.0390805
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184442, 0.0186017
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0298243, 0.0294991
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0728938
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361471, upper bound: 0.0365137
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361583, upper bound: 0.0365135
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155560, 0.0155305
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388863, 0.0390754
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184451, 0.0185979
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0298159, 0.0295009
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0728992
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361472, upper bound: 0.0365137
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361580, upper bound: 0.0365135
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155141, 0.0155452
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105190, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389896, 0.0387711
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0185228, 0.0183585
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0293133, 0.0296694
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0723616, 0.0733817
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362237, upper bound: 0.0365064
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362263, upper bound: 0.0365034
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155133, 0.0155446
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105154, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389854, 0.0387656
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0185197, 0.0183544
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0293043, 0.0296625
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0723359, 0.0733620
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362232, upper bound: 0.0365065
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362253, upper bound: 0.0365034
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155559, 0.0155325
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0389001, 0.0390743
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184555, 0.0185970
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0298141, 0.0295235
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0729637
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362675, upper bound: 0.0364925
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362697, upper bound: 0.0364903
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155551, 0.0155322
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388983, 0.0390688
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184542, 0.0185929
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0298051, 0.0295206
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0729555
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362677, upper bound: 0.0364926
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362707, upper bound: 0.0364903
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155592, 0.0155274
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388646, 0.0390809
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184288, 0.0185889
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0298116, 0.0294656
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0727980
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364903, upper bound: 0.0362707
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364926, upper bound: 0.0362677
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155584, 0.0155275
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388650, 0.0390757
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184291, 0.0185851
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0298033, 0.0294663
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0728000
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364903, upper bound: 0.0362697
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364925, upper bound: 0.0362675
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156009, 0.0155133
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105154
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0387656, 0.0393867
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0183544, 0.0188282
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0303122, 0.0293043
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0723359
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365034, upper bound: 0.0362253
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365065, upper bound: 0.0362232
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156002, 0.0155141
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105190
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0387711, 0.0393816
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0183585, 0.0188244
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0303039, 0.0293133
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0723616
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365034, upper bound: 0.0362263
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365064, upper bound: 0.0362237
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155583, 0.0155241
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388416, 0.0390746
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184116, 0.0185842
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0298014, 0.0294283
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0726909
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365135, upper bound: 0.0361580
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365137, upper bound: 0.0361472
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155575, 0.0155236
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105261
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0388380, 0.0390691
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0184088, 0.0185801
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0297925, 0.0294223
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0726740
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365135, upper bound: 0.0361583
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365137, upper bound: 0.0361471
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0156000, 0.0155142
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105198
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0387722, 0.0393805
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0183594, 0.0188235
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0303021, 0.0293151
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0723668
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365294, upper bound: 0.0361181
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365293, upper bound: 0.0361092
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155992, 0.0155150
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0105232
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0387774, 0.0393750
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0183632, 0.0188194
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0302931, 0.0293235
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0734371, 0.0723908
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365294, upper bound: 0.0361185
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365292, upper bound: 0.0361097
time: 1.09 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361097, upper bound: 0.0365292
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361185, upper bound: 0.0365294
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361092, upper bound: 0.0365293
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361181, upper bound: 0.0365294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361471, upper bound: 0.0365137
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361583, upper bound: 0.0365135
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361472, upper bound: 0.0365137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0361580, upper bound: 0.0365135
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362237, upper bound: 0.0365064
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362263, upper bound: 0.0365034
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362232, upper bound: 0.0365065
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362253, upper bound: 0.0365034
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362675, upper bound: 0.0364925
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362697, upper bound: 0.0364903
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362677, upper bound: 0.0364926
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0362707, upper bound: 0.0364903
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0364903, upper bound: 0.0362707
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0364926, upper bound: 0.0362677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0364903, upper bound: 0.0362697
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0364925, upper bound: 0.0362675
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365034, upper bound: 0.0362253
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365065, upper bound: 0.0362232
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365034, upper bound: 0.0362263
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365064, upper bound: 0.0362237
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365135, upper bound: 0.0361580
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365137, upper bound: 0.0361472
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365135, upper bound: 0.0361583
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365137, upper bound: 0.0361471
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365294, upper bound: 0.0361181
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365293, upper bound: 0.0361092
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365294, upper bound: 0.0361185
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 8, lower bound: -0.0365292, upper bound: 0.0361097

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154693, 0.0155062
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102666, 0.0104394
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0386007, 0.0383420
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0182202, 0.0180257
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286364, 0.0290580
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0700990, 0.0713071
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359750, upper bound: 0.0364227
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359599, upper bound: 0.0364438
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154708, 0.0155000
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102735, 0.0104101
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0385568, 0.0383523
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181872, 0.0180334
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286533, 0.0289865
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0701474, 0.0711022
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359883, upper bound: 0.0364216
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359754, upper bound: 0.0364433
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154686, 0.0155052
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102632, 0.0104346
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0385935, 0.0383368
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0182148, 0.0180218
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286280, 0.0290463
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0700751, 0.0712734
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359739, upper bound: 0.0364229
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359578, upper bound: 0.0364437
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154718, 0.0154999
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102784, 0.0104098
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0385564, 0.0383596
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181869, 0.0180390
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286652, 0.0289858
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0701815, 0.0711001
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359881, upper bound: 0.0364216
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359722, upper bound: 0.0364432
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155110, 0.0154877
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104404, 0.0103527
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384708, 0.0386398
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181226, 0.0182616
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291314, 0.0288464
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0713775, 0.0707007
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360263, upper bound: 0.0363877
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360175, upper bound: 0.0364203
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155125, 0.0154847
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104473, 0.0103386
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384497, 0.0386501
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181067, 0.0182694
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291483, 0.0288120
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714259, 0.0706021
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360389, upper bound: 0.0363850
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360307, upper bound: 0.0364184
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155103, 0.0154863
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104370, 0.0103463
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384612, 0.0386346
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181154, 0.0182577
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291230, 0.0288308
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0713536, 0.0706559
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360261, upper bound: 0.0363880
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360166, upper bound: 0.0364201
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155135, 0.0154848
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104522, 0.0103393
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384509, 0.0386575
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181076, 0.0182749
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291602, 0.0288139
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714601, 0.0706075
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360387, upper bound: 0.0363855
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360301, upper bound: 0.0364182
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154684, 0.0155058
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102625, 0.0104375
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0385978, 0.0383357
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0182181, 0.0180210
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286262, 0.0290534
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0700699, 0.0712937
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361030, upper bound: 0.0364003
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360768, upper bound: 0.0364175
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154695, 0.0154996
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102675, 0.0104084
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0385542, 0.0383433
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181853, 0.0180266
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286385, 0.0289823
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0701051, 0.0710900
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361059, upper bound: 0.0363913
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360774, upper bound: 0.0364108
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154676, 0.0155050
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102588, 0.0104338
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0385922, 0.0383302
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0182139, 0.0180168
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286172, 0.0290442
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0700442, 0.0712675
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361019, upper bound: 0.0364010
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360729, upper bound: 0.0364166
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154698, 0.0154990
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0102687, 0.0104055
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0385500, 0.0383451
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181821, 0.0180280
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286415, 0.0289754
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0701137, 0.0710703
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361059, upper bound: 0.0363922
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360757, upper bound: 0.0364108
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155101, 0.0154911
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104362, 0.0103686
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384947, 0.0386335
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181405, 0.0182569
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291212, 0.0288853
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0713484, 0.0708122
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361608, upper bound: 0.0363624
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361444, upper bound: 0.0363949
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155112, 0.0154868
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104413, 0.0103486
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384647, 0.0386411
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181179, 0.0182625
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291335, 0.0288364
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0713836, 0.0706720
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361639, upper bound: 0.0363573
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361467, upper bound: 0.0363922
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155093, 0.0154894
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104326, 0.0103608
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384829, 0.0386280
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181317, 0.0182527
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291122, 0.0288661
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0713227, 0.0707572
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361611, upper bound: 0.0363625
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361444, upper bound: 0.0363942
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155114, 0.0154866
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104425, 0.0103474
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384629, 0.0386429
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181166, 0.0182639
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291365, 0.0288335
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0713922, 0.0706637
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361649, upper bound: 0.0363574
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361474, upper bound: 0.0363922
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155130, 0.0154871
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104153, 0.0103501
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384670, 0.0386478
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181197, 0.0182455
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291178, 0.0288401
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714363, 0.0706827
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363922, upper bound: 0.0361474
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363574, upper bound: 0.0361649
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155145, 0.0154818
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104222, 0.0103249
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384292, 0.0386582
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180913, 0.0182533
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291347, 0.0287785
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714847, 0.0705062
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363942, upper bound: 0.0361444
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363625, upper bound: 0.0361611
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155123, 0.0154867
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104119, 0.0103480
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384639, 0.0386427
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181174, 0.0182417
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291095, 0.0288351
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714124, 0.0706683
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363922, upper bound: 0.0361467
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363573, upper bound: 0.0361639
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155156, 0.0154818
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104271, 0.0103252
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384296, 0.0386655
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180916, 0.0182588
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291467, 0.0287792
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0715188, 0.0705082
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363949, upper bound: 0.0361444
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363624, upper bound: 0.0361608
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155546, 0.0154698
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102687
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383451, 0.0389453
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180280, 0.0184826
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0296108, 0.0286415
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0726917, 0.0701137
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364108, upper bound: 0.0360757
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363922, upper bound: 0.0361059
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155561, 0.0154676
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102588
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383302, 0.0389557
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180168, 0.0184904
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0296277, 0.0286173
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0727401, 0.0700442
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364166, upper bound: 0.0360729
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364010, upper bound: 0.0361019
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155539, 0.0154695
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102675
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383432, 0.0389402
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180266, 0.0184788
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0296025, 0.0286385
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0726678, 0.0701051
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364108, upper bound: 0.0360774
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363913, upper bound: 0.0361059
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155571, 0.0154684
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102625
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383357, 0.0389630
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180210, 0.0184959
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0296397, 0.0286262
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0727742, 0.0700699
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364175, upper bound: 0.0360768
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364003, upper bound: 0.0361030
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155122, 0.0154856
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104111, 0.0103426
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384558, 0.0386416
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181113, 0.0182408
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291077, 0.0288219
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714072, 0.0706305
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364182, upper bound: 0.0360301
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363855, upper bound: 0.0360387
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155132, 0.0154785
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104162, 0.0103096
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384062, 0.0386491
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180740, 0.0182465
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291200, 0.0287412
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714424, 0.0703992
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364201, upper bound: 0.0360166
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363880, upper bound: 0.0360261
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155114, 0.0154845
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104075, 0.0103375
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384481, 0.0386360
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0181055, 0.0182367
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0290987, 0.0288094
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0713814, 0.0705946
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364184, upper bound: 0.0360307
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363850, upper bound: 0.0360389
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155135, 0.0154780
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0104174, 0.0103071
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0384026, 0.0386509
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180713, 0.0182479
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0291230, 0.0287353
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0714510, 0.0703823
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364203, upper bound: 0.0360175
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363877, upper bound: 0.0360263
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155537, 0.0154718
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102784
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383596, 0.0389391
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180390, 0.0184779
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0296007, 0.0286652
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0726626, 0.0701816
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364432, upper bound: 0.0359722
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364216, upper bound: 0.0359881
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155548, 0.0154686
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102632
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383368, 0.0389466
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180218, 0.0184836
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0296130, 0.0286280
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0726978, 0.0700751
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364437, upper bound: 0.0359578
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364229, upper bound: 0.0359739
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155530, 0.0154708
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102735
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383523, 0.0389336
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180334, 0.0184738
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0295917, 0.0286533
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0726368, 0.0701474
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364433, upper bound: 0.0359754
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364216, upper bound: 0.0359883
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155551, 0.0154693
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0105261, 0.0102666
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0383419, 0.0389484
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0180257, 0.0184850
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0296160, 0.0286364
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0727064, 0.0700990
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364438, upper bound: 0.0359599
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364227, upper bound: 0.0359750
time: 1.01 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359750, upper bound: 0.0364227
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359599, upper bound: 0.0364438
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359883, upper bound: 0.0364216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359754, upper bound: 0.0364433
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359739, upper bound: 0.0364229
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359578, upper bound: 0.0364437
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359881, upper bound: 0.0364216
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0359722, upper bound: 0.0364432
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360263, upper bound: 0.0363877
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360175, upper bound: 0.0364203
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360389, upper bound: 0.0363850
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360307, upper bound: 0.0364184
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360261, upper bound: 0.0363880
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360166, upper bound: 0.0364201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360387, upper bound: 0.0363855
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360301, upper bound: 0.0364182
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361030, upper bound: 0.0364003
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360768, upper bound: 0.0364175
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361059, upper bound: 0.0363913
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360774, upper bound: 0.0364108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361019, upper bound: 0.0364010
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360729, upper bound: 0.0364166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361059, upper bound: 0.0363922
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0360757, upper bound: 0.0364108
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361608, upper bound: 0.0363624
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361444, upper bound: 0.0363949
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361639, upper bound: 0.0363573
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361467, upper bound: 0.0363922
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361611, upper bound: 0.0363625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361444, upper bound: 0.0363942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361649, upper bound: 0.0363574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0361474, upper bound: 0.0363922
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363922, upper bound: 0.0361474
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363574, upper bound: 0.0361649
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363942, upper bound: 0.0361444
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363625, upper bound: 0.0361611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363922, upper bound: 0.0361467
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363573, upper bound: 0.0361639
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363949, upper bound: 0.0361444
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363624, upper bound: 0.0361608
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364108, upper bound: 0.0360757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363922, upper bound: 0.0361059
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364166, upper bound: 0.0360729
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364010, upper bound: 0.0361019
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364108, upper bound: 0.0360774
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363913, upper bound: 0.0361059
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364175, upper bound: 0.0360768
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364003, upper bound: 0.0361030
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364182, upper bound: 0.0360301
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363855, upper bound: 0.0360387
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364201, upper bound: 0.0360166
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363880, upper bound: 0.0360261
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364184, upper bound: 0.0360307
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363850, upper bound: 0.0360389
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364203, upper bound: 0.0360175
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0363877, upper bound: 0.0360263
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364432, upper bound: 0.0359722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364216, upper bound: 0.0359881
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364437, upper bound: 0.0359578
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364229, upper bound: 0.0359739
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364433, upper bound: 0.0359754
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364216, upper bound: 0.0359883
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364438, upper bound: 0.0359599
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 8, lower bound: -0.0364227, upper bound: 0.0359750

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154344, 0.0154768
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098355, 0.0100337
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381245, 0.0378277
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0178110, 0.0175879
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342503
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276676, 0.0281512
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0657521, 0.0671378
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354598, upper bound: 0.0360380
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354504, upper bound: 0.0360632
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154398, 0.0154753
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098609, 0.0100270
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381144, 0.0378657
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0178035, 0.0176165
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342767
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277295, 0.0281349
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659297, 0.0670910
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354462, upper bound: 0.0360673
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354363, upper bound: 0.0360953
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154370, 0.0154705
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098477, 0.0100044
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380806, 0.0378460
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177780, 0.0176016
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342630
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276974, 0.0280797
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0658377, 0.0669329
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354905, upper bound: 0.0360241
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354749, upper bound: 0.0360558
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154413, 0.0154715
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098678, 0.0100091
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380876, 0.0378761
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177833, 0.0176243
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342838
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277464, 0.0280911
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659781, 0.0669657
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354700, upper bound: 0.0360519
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354617, upper bound: 0.0360877
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154351, 0.0154757
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098385, 0.0100289
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381173, 0.0378322
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0178056, 0.0175913
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342534
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276749, 0.0281394
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0657733, 0.0671041
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354598, upper bound: 0.0360381
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354506, upper bound: 0.0360650
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154391, 0.0154743
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098575, 0.0100223
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381074, 0.0378606
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177982, 0.0176126
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342731
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277212, 0.0281235
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659058, 0.0670584
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354460, upper bound: 0.0360639
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354363, upper bound: 0.0360949
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154389, 0.0154704
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098566, 0.0100041
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380801, 0.0378593
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177777, 0.0176116
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342722
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277191, 0.0280790
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0658998, 0.0669308
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354905, upper bound: 0.0360263
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354749, upper bound: 0.0360580
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154424, 0.0154708
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098727, 0.0100059
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380829, 0.0378834
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177798, 0.0176298
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342889
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277584, 0.0280834
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0660123, 0.0669436
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354709, upper bound: 0.0360536
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354624, upper bound: 0.0360881
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154765, 0.0154582
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099993, 0.0099470
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379946, 0.0381305
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177134, 0.0178228
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281673, 0.0279395
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672733, 0.0665314
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355691, upper bound: 0.0359239
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355589, upper bound: 0.0359706
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154819, 0.0154566
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100247, 0.0099394
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379833, 0.0381685
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177049, 0.0178514
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282293, 0.0279212
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0674510, 0.0664788
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355614, upper bound: 0.0359846
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355513, upper bound: 0.0360404
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154791, 0.0154552
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100116, 0.0099328
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379735, 0.0381488
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176975, 0.0178366
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281971, 0.0279051
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0673589, 0.0664328
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355972, upper bound: 0.0359142
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355842, upper bound: 0.0359606
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154834, 0.0154554
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100316, 0.0099338
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379749, 0.0381789
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176986, 0.0178592
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282461, 0.0279075
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0674993, 0.0664396
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355854, upper bound: 0.0359741
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355770, upper bound: 0.0360354
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154771, 0.0154569
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100023, 0.0099405
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379850, 0.0381350
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177062, 0.0178262
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281746, 0.0279239
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672945, 0.0664866
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355637, upper bound: 0.0359238
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355484, upper bound: 0.0359707
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154812, 0.0154551
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100213, 0.0099322
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379725, 0.0381634
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176968, 0.0178476
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282209, 0.0279036
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0674270, 0.0664283
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355523, upper bound: 0.0359787
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355385, upper bound: 0.0360400
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154810, 0.0154554
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100204, 0.0099336
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379746, 0.0381621
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176984, 0.0178466
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282188, 0.0279070
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0674210, 0.0664382
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355942, upper bound: 0.0359143
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355829, upper bound: 0.0359608
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154844, 0.0154547
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100365, 0.0099305
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379700, 0.0381862
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176949, 0.0178647
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282581, 0.0278995
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0675335, 0.0664166
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355796, upper bound: 0.0359730
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355691, upper bound: 0.0360353
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154377, 0.0154763
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098510, 0.0100318
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381216, 0.0378508
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0178089, 0.0176053
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342663
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277053, 0.0281465
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0658603, 0.0671244
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356518, upper bound: 0.0360142
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356129, upper bound: 0.0360362
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154390, 0.0154747
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098567, 0.0100240
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381100, 0.0378595
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0178001, 0.0176118
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342723
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277194, 0.0281275
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659006, 0.0670700
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356107, upper bound: 0.0360332
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355750, upper bound: 0.0360624
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154357, 0.0154701
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098415, 0.0100026
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380780, 0.0378366
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177761, 0.0175946
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342565
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276822, 0.0280754
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0657940, 0.0669207
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356590, upper bound: 0.0359921
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356242, upper bound: 0.0360186
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154400, 0.0154701
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098618, 0.0100027
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380781, 0.0378670
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177762, 0.0176174
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342776
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277317, 0.0280756
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659358, 0.0669213
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356273, upper bound: 0.0360075
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355953, upper bound: 0.0360442
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154382, 0.0154755
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098532, 0.0100280
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381160, 0.0378542
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0178047, 0.0176078
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342687
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277108, 0.0281374
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0658760, 0.0670983
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356499, upper bound: 0.0360136
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356124, upper bound: 0.0360363
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154382, 0.0154734
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098531, 0.0100182
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0381013, 0.0378539
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177936, 0.0176076
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342685
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277104, 0.0281134
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0658749, 0.0670296
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356064, upper bound: 0.0360327
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355727, upper bound: 0.0360611
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154368, 0.0154695
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098468, 0.0099998
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380737, 0.0378446
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177729, 0.0176006
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342620
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276952, 0.0280686
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0658314, 0.0669010
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356569, upper bound: 0.0359927
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356191, upper bound: 0.0360211
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154403, 0.0154692
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098630, 0.0099986
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380719, 0.0378688
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177715, 0.0176188
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0342788
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277347, 0.0280655
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659444, 0.0668922
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356273, upper bound: 0.0360062
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355953, upper bound: 0.0360442
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154798, 0.0154616
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100148, 0.0099629
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380185, 0.0381537
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177313, 0.0178402
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282050, 0.0279785
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0673816, 0.0666429
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357800, upper bound: 0.0359146
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357515, upper bound: 0.0359477
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154810, 0.0154605
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100206, 0.0099578
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380108, 0.0381623
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177256, 0.0178467
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282191, 0.0279660
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0674218, 0.0666073
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357564, upper bound: 0.0359635
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357348, upper bound: 0.0360085
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154777, 0.0154573
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100053, 0.0099428
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379884, 0.0381395
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177088, 0.0178296
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281819, 0.0279295
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0673152, 0.0665027
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357858, upper bound: 0.0359034
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357578, upper bound: 0.0359386
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154821, 0.0154578
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100256, 0.0099450
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379917, 0.0381698
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177112, 0.0178524
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282314, 0.0279349
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0674570, 0.0665179
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357683, upper bound: 0.0359445
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357470, upper bound: 0.0359971
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154803, 0.0154600
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100170, 0.0099550
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0380067, 0.0381570
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177225, 0.0178428
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282105, 0.0279593
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0673972, 0.0665879
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357802, upper bound: 0.0359139
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357513, upper bound: 0.0359479
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154802, 0.0154585
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100169, 0.0099480
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379962, 0.0381568
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177146, 0.0178426
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282101, 0.0279422
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0673961, 0.0665389
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357534, upper bound: 0.0359539
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357349, upper bound: 0.0360054
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154789, 0.0154571
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100107, 0.0099417
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379867, 0.0381475
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177074, 0.0178356
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281949, 0.0279267
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0673526, 0.0664944
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357861, upper bound: 0.0359023
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357569, upper bound: 0.0359389
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154823, 0.0154566
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100268, 0.0099391
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379829, 0.0381717
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177046, 0.0178538
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282344, 0.0279205
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0674656, 0.0664768
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357668, upper bound: 0.0359426
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357464, upper bound: 0.0359966
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154777, 0.0154577
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099636, 0.0099444
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379907, 0.0381353
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177105, 0.0178141
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281547, 0.0279333
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0670696, 0.0665134
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359966, upper bound: 0.0357464
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359426, upper bound: 0.0357668
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154832, 0.0154547
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099891, 0.0099304
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379699, 0.0381733
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176948, 0.0178427
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282167, 0.0278993
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672472, 0.0664160
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359389, upper bound: 0.0357569
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359023, upper bound: 0.0357861
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154803, 0.0154523
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099759, 0.0099191
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379529, 0.0381536
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176821, 0.0178279
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281846, 0.0278717
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0671552, 0.0663370
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360054, upper bound: 0.0357349
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359539, upper bound: 0.0357534
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154846, 0.0154534
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099960, 0.0099243
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379606, 0.0381837
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176878, 0.0178505
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282336, 0.0278842
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672956, 0.0663729
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359479, upper bound: 0.0357513
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359139, upper bound: 0.0357802
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154784, 0.0154572
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099667, 0.0099423
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379876, 0.0381398
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177082, 0.0178175
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281621, 0.0279283
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0670907, 0.0664990
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359971, upper bound: 0.0357470
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359445, upper bound: 0.0357683
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154824, 0.0154536
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099856, 0.0099252
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379620, 0.0381682
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176889, 0.0178389
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282084, 0.0278865
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672233, 0.0663795
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359386, upper bound: 0.0357578
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359034, upper bound: 0.0357858
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154822, 0.0154523
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099848, 0.0099194
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379534, 0.0381669
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176824, 0.0178379
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282063, 0.0278724
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672173, 0.0663390
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360085, upper bound: 0.0357348
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359635, upper bound: 0.0357564
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154857, 0.0154528
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0100009, 0.0099214
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379563, 0.0381910
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176846, 0.0178560
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282455, 0.0278771
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0673298, 0.0663525
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359477, upper bound: 0.0357515
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359146, upper bound: 0.0357800
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155197, 0.0154403
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101324, 0.0098630
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378688, 0.0384350
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176188, 0.0180490
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342788, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286561, 0.0277347
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0685586, 0.0659444
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360442, upper bound: 0.0355953
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360062, upper bound: 0.0356273
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155251, 0.0154368
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101578, 0.0098468
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378446, 0.0384730
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176006, 0.0180776
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342620, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287181, 0.0276952
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0687362, 0.0658314
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360211, upper bound: 0.0356191
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359927, upper bound: 0.0356569
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155223, 0.0154382
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101447, 0.0098531
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378539, 0.0384533
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176076, 0.0180627
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342685, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286860, 0.0277104
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0686442, 0.0658749
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360611, upper bound: 0.0355727
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360327, upper bound: 0.0356064
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155266, 0.0154382
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101648, 0.0098532
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378542, 0.0384834
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176078, 0.0180854
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342687, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287350, 0.0277108
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0687846, 0.0658760
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360363, upper bound: 0.0356124
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360136, upper bound: 0.0356499
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155203, 0.0154400
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101355, 0.0098618
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378670, 0.0384395
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176174, 0.0180524
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342776, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286635, 0.0277317
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0685797, 0.0659358
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360442, upper bound: 0.0355953
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360075, upper bound: 0.0356273
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155244, 0.0154357
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101544, 0.0098415
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378366, 0.0384679
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175946, 0.0180737
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342565, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287097, 0.0276822
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0687123, 0.0657940
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360186, upper bound: 0.0356242
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359921, upper bound: 0.0356590
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155242, 0.0154390
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101536, 0.0098567
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378595, 0.0384666
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176118, 0.0180727
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342723, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287076, 0.0277194
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0687063, 0.0659006
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360624, upper bound: 0.0355750
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360332, upper bound: 0.0356107
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155276, 0.0154377
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101696, 0.0098510
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378508, 0.0384907
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176053, 0.0180909
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342663, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287469, 0.0277053
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0688188, 0.0658603
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360362, upper bound: 0.0356129
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360142, upper bound: 0.0356518
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154810, 0.0154561
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099791, 0.0099369
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379796, 0.0381584
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0177021, 0.0178316
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281925, 0.0279151
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0671778, 0.0664612
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360353, upper bound: 0.0355691
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359730, upper bound: 0.0355796
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154823, 0.0154527
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099849, 0.0099210
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379558, 0.0381671
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176842, 0.0178380
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282065, 0.0278763
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672181, 0.0663503
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359608, upper bound: 0.0355829
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359143, upper bound: 0.0355942
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154790, 0.0154490
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099696, 0.0099038
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379300, 0.0381442
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176648, 0.0178209
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281693, 0.0278343
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0671115, 0.0662299
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360400, upper bound: 0.0355385
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359787, upper bound: 0.0355523
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154833, 0.0154482
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099899, 0.0098998
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379240, 0.0381746
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176603, 0.0178437
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282188, 0.0278246
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672533, 0.0662020
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359707, upper bound: 0.0355484
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359238, upper bound: 0.0355637
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154815, 0.0154550
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099814, 0.0099318
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379718, 0.0381618
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176963, 0.0178341
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281980, 0.0279025
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0671935, 0.0664253
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360354, upper bound: 0.0355770
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359741, upper bound: 0.0355854
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154815, 0.0154518
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099812, 0.0099167
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379492, 0.0381616
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176793, 0.0178339
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281975, 0.0278656
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0671924, 0.0663196
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359606, upper bound: 0.0355842
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359142, upper bound: 0.0355972
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154801, 0.0154485
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099750, 0.0099014
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379264, 0.0381522
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176621, 0.0178269
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343136, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0281824, 0.0278284
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0671489, 0.0662130
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360404, upper bound: 0.0355513
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359846, upper bound: 0.0355614
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154836, 0.0154471
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0099912, 0.0098947
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0379164, 0.0381764
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176546, 0.0178451
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0343118, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0282218, 0.0278121
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0672619, 0.0661663
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359706, upper bound: 0.0355589
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359239, upper bound: 0.0355691
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155230, 0.0154424
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101479, 0.0098727
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378834, 0.0384581
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176298, 0.0180664
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342889, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286939, 0.0277584
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0686668, 0.0660123
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360881, upper bound: 0.0354624
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360536, upper bound: 0.0354709
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155242, 0.0154389
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101537, 0.0098566
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378593, 0.0384668
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176116, 0.0180729
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342722, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287079, 0.0277191
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0687071, 0.0658998
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360580, upper bound: 0.0354749
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360263, upper bound: 0.0354905
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155210, 0.0154391
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101384, 0.0098575
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378606, 0.0384439
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176126, 0.0180557
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342731, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286707, 0.0277212
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0686004, 0.0659058
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360949, upper bound: 0.0354363
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360639, upper bound: 0.0354460
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155253, 0.0154351
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101587, 0.0098385
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378322, 0.0384743
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175913, 0.0180785
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342534, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287202, 0.0276749
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0687423, 0.0657733
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360650, upper bound: 0.0354506
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360381, upper bound: 0.0354598
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155235, 0.0154413
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101502, 0.0098678
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378761, 0.0384615
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176243, 0.0180689
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342838, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286993, 0.0277464
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0686825, 0.0659781
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360877, upper bound: 0.0354617
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360519, upper bound: 0.0354700
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155234, 0.0154370
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101500, 0.0098477
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378460, 0.0384613
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176016, 0.0180687
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342630, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286989, 0.0276974
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0686814, 0.0658377
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360558, upper bound: 0.0354749
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360241, upper bound: 0.0354905
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155221, 0.0154398
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101438, 0.0098609
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378657, 0.0384519
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0176165, 0.0180617
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342766, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0286838, 0.0277296
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0686379, 0.0659297
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360953, upper bound: 0.0354363
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360673, upper bound: 0.0354462
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0155256, 0.0154344
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0101599, 0.0098355
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378277, 0.0384761
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175879, 0.0180799
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342503, 0.0343136
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0287232, 0.0276676
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0687509, 0.0657521
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360632, upper bound: 0.0354504
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360380, upper bound: 0.0354598
time: 1.07 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354598, upper bound: 0.0360380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354504, upper bound: 0.0360632
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354462, upper bound: 0.0360673
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354363, upper bound: 0.0360953
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354905, upper bound: 0.0360241
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354749, upper bound: 0.0360558
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354700, upper bound: 0.0360519
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354617, upper bound: 0.0360877
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354598, upper bound: 0.0360381
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354506, upper bound: 0.0360650
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354460, upper bound: 0.0360639
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354363, upper bound: 0.0360949
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354905, upper bound: 0.0360263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354749, upper bound: 0.0360580
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354709, upper bound: 0.0360536
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0354624, upper bound: 0.0360881
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355691, upper bound: 0.0359239
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355589, upper bound: 0.0359706
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355614, upper bound: 0.0359846
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355513, upper bound: 0.0360404
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355972, upper bound: 0.0359142
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355842, upper bound: 0.0359606
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355854, upper bound: 0.0359741
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355770, upper bound: 0.0360354
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355637, upper bound: 0.0359238
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355484, upper bound: 0.0359707
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355523, upper bound: 0.0359787
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355385, upper bound: 0.0360400
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355942, upper bound: 0.0359143
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355829, upper bound: 0.0359608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355796, upper bound: 0.0359730
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355691, upper bound: 0.0360353
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356518, upper bound: 0.0360142
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356129, upper bound: 0.0360362
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356107, upper bound: 0.0360332
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355750, upper bound: 0.0360624
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356590, upper bound: 0.0359921
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356242, upper bound: 0.0360186
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356273, upper bound: 0.0360075
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355953, upper bound: 0.0360442
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356499, upper bound: 0.0360136
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356124, upper bound: 0.0360363
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356064, upper bound: 0.0360327
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355727, upper bound: 0.0360611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356569, upper bound: 0.0359927
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356191, upper bound: 0.0360211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0356273, upper bound: 0.0360062
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0355953, upper bound: 0.0360442
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357800, upper bound: 0.0359146
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357515, upper bound: 0.0359477
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357564, upper bound: 0.0359635
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357348, upper bound: 0.0360085
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357858, upper bound: 0.0359034
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357578, upper bound: 0.0359386
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357683, upper bound: 0.0359445
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357470, upper bound: 0.0359971
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357802, upper bound: 0.0359139
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357513, upper bound: 0.0359479
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357534, upper bound: 0.0359539
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357349, upper bound: 0.0360054
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357861, upper bound: 0.0359023
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357569, upper bound: 0.0359389
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357668, upper bound: 0.0359426
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0357464, upper bound: 0.0359966
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359966, upper bound: 0.0357464
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359426, upper bound: 0.0357668
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359389, upper bound: 0.0357569
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359023, upper bound: 0.0357861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360054, upper bound: 0.0357349
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359539, upper bound: 0.0357534
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359479, upper bound: 0.0357513
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359139, upper bound: 0.0357802
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359971, upper bound: 0.0357470
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359445, upper bound: 0.0357683
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359386, upper bound: 0.0357578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359034, upper bound: 0.0357858
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360085, upper bound: 0.0357348
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359635, upper bound: 0.0357564
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359477, upper bound: 0.0357515
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359146, upper bound: 0.0357800
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360442, upper bound: 0.0355953
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360062, upper bound: 0.0356273
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360211, upper bound: 0.0356191
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359927, upper bound: 0.0356569
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360611, upper bound: 0.0355727
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360327, upper bound: 0.0356064
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360363, upper bound: 0.0356124
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360136, upper bound: 0.0356499
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360442, upper bound: 0.0355953
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360075, upper bound: 0.0356273
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360186, upper bound: 0.0356242
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359921, upper bound: 0.0356590
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360624, upper bound: 0.0355750
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360332, upper bound: 0.0356107
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360362, upper bound: 0.0356129
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360142, upper bound: 0.0356518
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360353, upper bound: 0.0355691
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359730, upper bound: 0.0355796
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359608, upper bound: 0.0355829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359143, upper bound: 0.0355942
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360400, upper bound: 0.0355385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359787, upper bound: 0.0355523
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359707, upper bound: 0.0355484
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359238, upper bound: 0.0355637
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360354, upper bound: 0.0355770
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359741, upper bound: 0.0355854
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359606, upper bound: 0.0355842
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359142, upper bound: 0.0355972
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360404, upper bound: 0.0355513
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359846, upper bound: 0.0355614
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359706, upper bound: 0.0355589
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0359239, upper bound: 0.0355691
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360881, upper bound: 0.0354624
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360536, upper bound: 0.0354709
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360580, upper bound: 0.0354749
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360263, upper bound: 0.0354905
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360949, upper bound: 0.0354363
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360639, upper bound: 0.0354460
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360650, upper bound: 0.0354506
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360381, upper bound: 0.0354598
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360877, upper bound: 0.0354617
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360519, upper bound: 0.0354700
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360558, upper bound: 0.0354749
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360241, upper bound: 0.0354905
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360953, upper bound: 0.0354363
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360673, upper bound: 0.0354462
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360632, upper bound: 0.0354504
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.65
Output dim: 8, lower bound: -0.0360380, upper bound: 0.0354598

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154010, 0.0154382
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096430, 0.0098169
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377998, 0.0375393
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175669, 0.0173710
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342310, 0.0340503
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271977, 0.0276222
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644060, 0.0656222
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348611, upper bound: 0.0352398
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348611, upper bound: 0.0352398
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0153958, 0.0154418
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096187, 0.0098338
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378251, 0.0375030
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175860, 0.0173437
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342485, 0.0340251
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271386, 0.0276634
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0642365, 0.0657403
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348581, upper bound: 0.0352619
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348581, upper bound: 0.0352619
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154038, 0.0154367
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096560, 0.0098102
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377898, 0.0375589
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175594, 0.0173858
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342240, 0.0340638
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272296, 0.0276059
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644974, 0.0655753
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348448, upper bound: 0.0352567
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348448, upper bound: 0.0352567
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154012, 0.0154425
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096441, 0.0098373
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378303, 0.0375411
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175898, 0.0173723
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342521, 0.0340515
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272006, 0.0276719
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644141, 0.0657644
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348383, upper bound: 0.0352804
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348383, upper bound: 0.0352804
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154031, 0.0154319
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096530, 0.0097876
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377559, 0.0375544
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175339, 0.0173824
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342005, 0.0340607
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272223, 0.0275507
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644764, 0.0654173
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348760, upper bound: 0.0352381
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348760, upper bound: 0.0352381
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0153984, 0.0154349
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096310, 0.0098015
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377767, 0.0375214
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175496, 0.0173575
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342149, 0.0340378
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271685, 0.0275846
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0643221, 0.0655143
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348677, upper bound: 0.0352606
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348677, upper bound: 0.0352606
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154061, 0.0154329
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096671, 0.0097923
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377630, 0.0375755
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175392, 0.0173982
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342054, 0.0340753
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272566, 0.0275622
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0645747, 0.0654501
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348567, upper bound: 0.0352523
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348567, upper bound: 0.0352523
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154027, 0.0154387
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096510, 0.0098195
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378037, 0.0375514
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175699, 0.0173801
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342337, 0.0340587
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272174, 0.0276286
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644625, 0.0656405
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348513, upper bound: 0.0352802
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348513, upper bound: 0.0352802
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154011, 0.0154371
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096436, 0.0098121
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377926, 0.0375403
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175615, 0.0173718
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342259, 0.0340510
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271994, 0.0276105
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644107, 0.0655885
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348610, upper bound: 0.0352416
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348610, upper bound: 0.0352416
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0153965, 0.0154404
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096217, 0.0098275
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378157, 0.0375075
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175788, 0.0173471
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342420, 0.0340282
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271460, 0.0276480
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0642577, 0.0656962
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348575, upper bound: 0.0352638
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348575, upper bound: 0.0352638
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154026, 0.0154357
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096504, 0.0098055
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377828, 0.0375505
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175541, 0.0173794
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342192, 0.0340580
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272160, 0.0275945
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644582, 0.0655427
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348441, upper bound: 0.0352561
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348441, upper bound: 0.0352561
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154005, 0.0154407
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096407, 0.0098288
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378176, 0.0375359
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175803, 0.0173685
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342433, 0.0340479
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271922, 0.0276512
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0643902, 0.0657053
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348371, upper bound: 0.0352802
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348371, upper bound: 0.0352802
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154045, 0.0154318
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096596, 0.0097873
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377555, 0.0375642
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175336, 0.0173898
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342002, 0.0340675
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272383, 0.0275500
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0645223, 0.0654152
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348762, upper bound: 0.0352405
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348762, upper bound: 0.0352405
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154003, 0.0154358
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096398, 0.0098059
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377834, 0.0375347
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175546, 0.0173675
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342195, 0.0340470
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271901, 0.0275954
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0643842, 0.0655454
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348679, upper bound: 0.0352633
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348679, upper bound: 0.0352633
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154064, 0.0154322
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096683, 0.0097891
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377582, 0.0375773
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175356, 0.0173996
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342021, 0.0340766
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272596, 0.0275544
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0645832, 0.0654280
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348558, upper bound: 0.0352524
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348558, upper bound: 0.0352524
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154038, 0.0154387
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096559, 0.0098194
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378035, 0.0375587
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175697, 0.0173856
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342335, 0.0340637
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272294, 0.0276282
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0644967, 0.0656393
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348508, upper bound: 0.0352801
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348508, upper bound: 0.0352801
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154430, 0.0154196
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098068, 0.0097302
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376699, 0.0378422
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174693, 0.0176060
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341409, 0.0342579
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276974, 0.0274106
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659273, 0.0650158
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349199, upper bound: 0.0351966
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349199, upper bound: 0.0351966
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154379, 0.0154234
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0097825, 0.0097479
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376964, 0.0378058
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174892, 0.0175787
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341592, 0.0342327
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276383, 0.0274537
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0657577, 0.0651394
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349188, upper bound: 0.0352254
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349188, upper bound: 0.0352254
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154458, 0.0154180
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098199, 0.0097227
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376587, 0.0378617
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174608, 0.0176207
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341331, 0.0342714
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277293, 0.0273922
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0660187, 0.0649632
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352217
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352217
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154433, 0.0154247
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098080, 0.0097541
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377057, 0.0378439
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174962, 0.0176073
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341657, 0.0342591
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277003, 0.0274689
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659354, 0.0651828
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352547
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352547
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154452, 0.0154166
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098169, 0.0097161
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376488, 0.0378572
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174534, 0.0176173
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341262, 0.0342683
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277220, 0.0273761
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659976, 0.0649172
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349352, upper bound: 0.0351961
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349352, upper bound: 0.0351961
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154405, 0.0154208
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0097948, 0.0097356
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376781, 0.0378242
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174754, 0.0175925
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341465, 0.0342454
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276682, 0.0274239
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0658433, 0.0650539
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349299, upper bound: 0.0352257
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349299, upper bound: 0.0352257
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154482, 0.0154168
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098309, 0.0097170
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376503, 0.0378783
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174545, 0.0176332
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341272, 0.0342829
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277563, 0.0273785
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0660959, 0.0649239
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349234, upper bound: 0.0352174
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349234, upper bound: 0.0352174
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154448, 0.0154234
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098149, 0.0097477
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376962, 0.0378543
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174890, 0.0176151
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341591, 0.0342662
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277172, 0.0274533
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659837, 0.0651383
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349221, upper bound: 0.0352542
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349221, upper bound: 0.0352542
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154432, 0.0154183
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098075, 0.0097238
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376603, 0.0378432
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174620, 0.0176068
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341342, 0.0342586
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276991, 0.0273949
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659319, 0.0649710
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349213, upper bound: 0.0351979
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349213, upper bound: 0.0351979
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154385, 0.0154232
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0097856, 0.0097469
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376950, 0.0378104
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174881, 0.0175821
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341582, 0.0342358
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276457, 0.0274513
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0657789, 0.0651326
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349201, upper bound: 0.0352254
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349201, upper bound: 0.0352254
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154446, 0.0154165
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098143, 0.0097154
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376479, 0.0378533
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174526, 0.0176144
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341255, 0.0342656
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277157, 0.0273746
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659794, 0.0649127
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352214
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352214
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154426, 0.0154234
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098045, 0.0097481
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376967, 0.0378388
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174894, 0.0176035
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341594, 0.0342555
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276919, 0.0274542
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659114, 0.0651408
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352541
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349081, upper bound: 0.0352541
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154466, 0.0154168
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098234, 0.0097168
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376500, 0.0378671
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174542, 0.0176247
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341270, 0.0342751
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277380, 0.0273780
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0660436, 0.0649226
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349352, upper bound: 0.0351980
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349352, upper bound: 0.0351980
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154424, 0.0154215
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098037, 0.0097389
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376830, 0.0378375
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174791, 0.0176025
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341499, 0.0342546
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0276898, 0.0274319
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0659054, 0.0650769
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349317, upper bound: 0.0352257
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349317, upper bound: 0.0352257
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154485, 0.0154161
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098321, 0.0097138
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376453, 0.0378801
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174508, 0.0176345
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341238, 0.0342842
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277593, 0.0273705
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0661044, 0.0649010
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349235, upper bound: 0.0352174
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349235, upper bound: 0.0352174
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154458, 0.0154234
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0098198, 0.0097481
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0376967, 0.0378616
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0174894, 0.0176206
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0341594, 0.0342713
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0277291, 0.0274542
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0660179, 0.0651408
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349224, upper bound: 0.0352536
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349224, upper bound: 0.0352536
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154054, 0.0154377
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096636, 0.0098150
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377970, 0.0375702
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175648, 0.0173943
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342290, 0.0340717
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272481, 0.0276175
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0645502, 0.0656088
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350149, upper bound: 0.0351994
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350149, upper bound: 0.0351994
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0153991, 0.0154414
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096342, 0.0098320
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378224, 0.0375262
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175839, 0.0173612
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342466, 0.0340412
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271763, 0.0276589
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0643447, 0.0657274
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349915, upper bound: 0.0352083
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349915, upper bound: 0.0352083
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154047, 0.0154361
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096603, 0.0098072
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0377853, 0.0375653
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175560, 0.0173905
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342209, 0.0340682
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0272400, 0.0275985
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0645271, 0.0655544
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349830, upper bound: 0.0352206
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349830, upper bound: 0.0352206
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0121024, 0.0041448, -0.0121024, 0.0041448, -0.0154003, 0.0154422
1: -0.0028523, 0.0076738, -0.0028523, 0.0076738, -0.0096400, 0.0098360
2: 0.0048477, 0.0452012, 0.0048477, 0.0452012, -0.0378284, 0.0375348
3: -0.0069817, 0.0127687, -0.0069817, 0.0127687, -0.0175884, 0.0173677
4: -0.0108195, 0.0234941, -0.0108195, 0.0234941, -0.0342508, 0.0340471
5: 0.0007365, 0.0121147, 0.0007365, 0.0121147, -0.0113782, 0.0113782
6: 0.0002402, 0.0125671, 0.0002402, 0.0125671, -0.0123269, 0.0123269
7: -0.0361363, -0.0005037, -0.0361363, -0.0005037, -0.0271904, 0.0276688
8: 0.9489108, 1.0223479, 0.9489108, 1.0223479, -0.0643850, 0.0657557
9: -0.0098589, 0.0099128, -0.0098589, 0.0099128, -0.0197717, 0.0197717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.64 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.60 + 596.59 = 600.19 seconds
